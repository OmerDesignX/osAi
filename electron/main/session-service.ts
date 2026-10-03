import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { closeSync, createReadStream, openSync } from "node:fs";
import {
  phaseProgress,
  recoverFineTuneProgress,
  type TrainingStage,
} from "./training-progress.js";
import {
  lossMetric,
  metricHeader,
  metricRow,
  parseMetricRow,
  type TrainingMetric,
} from "./training-metrics.js";
import type {
  BackendStatus,
  Preferences,
  SessionState,
  TrainingRequest,
  WorkerJob,
} from "../types.js";

const activeStatuses = new Set([
  "queued",
  "running",
  "pausing",
  "paused",
  "stopping",
]);
const BACKEND_STATUS_TIMEOUT_MS = 5_000;
const BACKEND_STATUS_OUTPUT_LIMIT = 8 * 1024;
const alignmentTypes = new Set([
  "auto",
  "dpo",
  "ipo",
  "simpo",
  "orpo",
  "cpo",
  "kto",
  "ppo",
  "reinforce",
  "rloo",
  "grpo",
]);
const targetModules = new Set([
  "self_attn.q_proj",
  "self_attn.k_proj",
  "self_attn.v_proj",
  "self_attn.o_proj",
  "mlp.gate_proj",
  "mlp.up_proj",
  "mlp.down_proj",
]);

export function osAiVersionFromOutput(output: string) {
  for (const rawLine of output.split(/\r?\n/)) {
    const line = rawLine.replace(/\s+/g, " ").trim();
    if (
      /^osai\s+v?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/i.test(
        line,
      )
    )
      return line.slice(0, 120);
  }
  return "";
}
const splitModes = new Set(["auto", "none", "layer", "row", "tensor"]);
const defaultModelsRoot = path.join(os.homedir(), "osAi", "models");

async function readProgressLog(file: string) {
  const handle = await fs.open(file, "r").catch(() => null);
  if (!handle) return "";
  try {
    const size = (await handle.stat()).size;
    const headLength = Math.min(size, 32 * 1024);
    const tailLength = Math.min(Math.max(0, size - headLength), 192 * 1024);
    const head = Buffer.alloc(headLength);
    const tail = Buffer.alloc(tailLength);
    await handle.read(head, 0, headLength, 0);
    if (tailLength)
      await handle.read(
        tail,
        0,
        tailLength,
        Math.max(headLength, size - tailLength),
      );
    return tailLength
      ? `${head.toString("utf8")}\n${tail.toString("utf8")}`
      : head.toString("utf8");
  } finally {
    await handle.close();
  }
}

async function recoverVisibleProgress(state: SessionState) {
  if (
    !activeStatuses.has(state.status) ||
    state.request?.stage === "alignment" ||
    ["alignment", "rollouts", "publishing", "complete"].includes(state.phase)
  )
    return state;
  const recovered = recoverFineTuneProgress(
    await readProgressLog(state.logPath),
    state.request?.iterations || 1,
  );
  if (!recovered || recovered.completed < 1) return state;
  const stage = (state.request?.stage || "fine-tuning") as TrainingStage;
  const progress = Math.min(
    99,
    Math.round(
      phaseProgress(recovered.completed, recovered.total, "fine-tuning", stage),
    ),
  );
  if (progress <= state.progress) return state;
  return {
    ...state,
    phase: "fine-tuning" as const,
    progress,
    trainingPercent: Math.min(
      100,
      (100 * recovered.completed) / recovered.total,
    ),
    indeterminate: false,
    message:
      state.status === "paused"
        ? state.message
        : `Fine-tuning update ${recovered.completed} of ${recovered.total}`,
  };
}

function cleanName(value: string) {
  return (
    value
      .trim()
      .replace(/[^A-Za-z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 64) || "training"
  );
}

function timestamp(date = new Date()) {
  return date.toISOString().replace(/[-:]/g, "").replace("T", "-").slice(0, 15);
}

function positiveInteger(value: number, label: string, maximum = 1_000_000) {
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum)
    throw new Error(`${label} must be between 1 and ${maximum}`);
  return value;
}

function optionalInteger(
  value: number | null | undefined,
  label: string,
  minimum = 1,
  maximum = 1_000_000,
) {
  if (value === null || value === undefined) return null;
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum)
    throw new Error(`${label} must be between ${minimum} and ${maximum}`);
  return value;
}

function optionalNumber(
  value: number | null | undefined,
  label: string,
  minimum: number,
  maximum = Number.MAX_VALUE,
  minimumInclusive = true,
) {
  if (value === null || value === undefined) return null;
  if (
    !Number.isFinite(value) ||
    (minimumInclusive ? value < minimum : value <= minimum) ||
    value > maximum
  )
    throw new Error(
      `${label} must be ${minimumInclusive ? "at least" : "greater than"} ${minimum}${maximum < Number.MAX_VALUE ? ` and at most ${maximum}` : ""}`,
    );
  return value;
}

function pushOptional(args: string[], flag: string, value: number | null) {
  if (value !== null) args.push(flag, String(value));
}

function tensorSplit(value: string | undefined) {
  const text = value?.trim() ?? "";
  if (!text) return "";
  const weights = text.split(",").map((item) => Number(item.trim()));
  if (
    weights.some((item) => !Number.isFinite(item) || item <= 0) ||
    weights.length > 64
  )
    throw new Error(
      "Tensor split must contain positive comma-separated weights",
    );
  return weights.join(",");
}

function deviceList(value: string | undefined) {
  if (!value?.trim()) return [];
  const devices = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (
    devices.length > 64 ||
    devices.some(
      (device) =>
        device.length > 128 || device.includes("\0") || /[\r\n]/.test(device),
    )
  )
    throw new Error("Device order contains an invalid device name");
  return devices;
}

async function assertDirectory(value: string, label: string) {
  if (!value || !path.isAbsolute(value))
    throw new Error(`${label} must be an absolute directory path`);
  const stat = await fs.stat(value).catch(() => null);
  if (!stat?.isDirectory()) throw new Error(`${label} does not exist`);
  return path.resolve(value);
}

async function assertExistingPath(value: string, label: string) {
  if (!value || !path.isAbsolute(value))
    throw new Error(`${label} must be an absolute path`);
  if (!(await fs.stat(value).catch(() => null)))
    throw new Error(`${label} does not exist`);
  return path.resolve(value);
}

async function ensureSessionsRoot(value: string) {
  if (!value || !path.isAbsolute(value))
    throw new Error("Session save location must be an absolute directory path");
  const root = path.resolve(value);
  await fs.mkdir(root, { recursive: true, mode: 0o700 });
  const stat = await fs.stat(root).catch(() => null);
  if (!stat?.isDirectory())
    throw new Error("Session save location is not a directory");
  return root;
}

const mediaKeys = new Set([
  "image",
  "images",
  "video",
  "videos",
  "audio",
  "audios",
]);

function localMediaPath(value: string, sourceDirectory: string) {
  const trimmed = value.trim();
  if (
    !trimmed ||
    trimmed.startsWith("//") ||
    /^[A-Za-z][A-Za-z0-9+.-]*:/.test(trimmed)
  )
    return value;
  return path.isAbsolute(trimmed)
    ? path.normalize(trimmed)
    : path.resolve(sourceDirectory, trimmed);
}

function resolveMediaValue(value: unknown, sourceDirectory: string): unknown {
  if (typeof value === "string") return localMediaPath(value, sourceDirectory);
  if (Array.isArray(value))
    return value.map((item) => resolveMediaValue(item, sourceDirectory));
  if (!value || typeof value !== "object") return value;
  const result = { ...(value as Record<string, unknown>) };
  if (typeof result.path === "string")
    result.path = localMediaPath(result.path, sourceDirectory);
  if (typeof result.url === "string")
    result.url = localMediaPath(result.url, sourceDirectory);
  return result;
}

export function absolutizeDatasetMedia(
  value: unknown,
  sourceDirectory: string,
): unknown {
  if (Array.isArray(value))
    return value.map((item) => absolutizeDatasetMedia(item, sourceDirectory));
  if (!value || typeof value !== "object") return value;
  const result = { ...(value as Record<string, unknown>) };
  const type = typeof result.type === "string" ? result.type.toLowerCase() : "";
  const modality = type.replace(/_url$/, "");
  if (["image", "video", "audio"].includes(modality)) {
    for (const key of [modality, `${modality}_url`, "url", "path"])
      if (key in result)
        result[key] = resolveMediaValue(result[key], sourceDirectory);
  }
  for (const [key, child] of Object.entries(result)) {
    result[key] = mediaKeys.has(key)
      ? resolveMediaValue(child, sourceDirectory)
      : absolutizeDatasetMedia(child, sourceDirectory);
  }
  return result;
}

export function formatTerminalOutput(value: string) {
  const withoutControls = value
    // Older Windows worker pipes may have decoded the native ± byte as U+FFFD.
    .replace(/\b(loss|acc)=([0-9.eE+-]+)\uFFFD(?=[0-9])/gi, "$1=$2±")
    .replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g, "")
    .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "")
    .replace(/\u0008/g, "");
  const rendered = withoutControls
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => {
      const updates = line.split("\r");
      return updates[updates.length - 1] || "";
    });
  const compact: string[] = [];
  for (const line of rendered) {
    if (line || compact.at(-1) !== "" || compact.at(-2) !== "")
      compact.push(line);
  }
  return compact.join("\n").trimEnd();
}

async function writeJsonLines(input: string, output: string) {
  const handle = await fs.open(output, "wx", 0o600);
  const lines = readline.createInterface({
    input: createReadStream(input, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });
  let rows = 0;
  let buffer = "";
  try {
    for await (const line of lines) {
      if (!line.trim()) continue;
      let row: unknown;
      try {
        row = JSON.parse(line);
      } catch {
        throw new Error(`Invalid JSON in dataset file: ${input}`);
      }
      if (!row || typeof row !== "object" || Array.isArray(row))
        throw new Error(`Dataset rows must be JSON objects: ${input}`);
      buffer += `${JSON.stringify(absolutizeDatasetMedia(row, path.dirname(input)))}\n`;
      rows += 1;
      if (buffer.length >= 1024 * 1024) {
        await handle.write(buffer);
        buffer = "";
      }
    }
    if (buffer) await handle.write(buffer);
  } finally {
    lines.close();
    await handle.close();
  }
  if (rows < 1) throw new Error(`Dataset file is empty: ${input}`);
}

async function writeJsonDocument(input: string, output: string) {
  const source = await fs.readFile(input, "utf8");
  let parsed: unknown;
  try {
    parsed = JSON.parse(source);
  } catch {
    await writeJsonLines(input, output);
    return;
  }
  const container = parsed as Record<string, unknown> | null;
  const records = Array.isArray(parsed)
    ? parsed
    : container && Array.isArray(container.train)
      ? container.train
      : container && Array.isArray(container.data)
        ? container.data
        : container && Array.isArray(container.records)
          ? container.records
          : container && Array.isArray(container.examples)
            ? container.examples
            : container && Array.isArray(container.items)
              ? container.items
              : [parsed];
  if (records.length < 1) throw new Error(`Dataset file is empty: ${input}`);
  const lines = records.map((row) => {
    if (!row || typeof row !== "object" || Array.isArray(row))
      throw new Error(`Dataset rows must be JSON objects: ${input}`);
    return JSON.stringify(absolutizeDatasetMedia(row, path.dirname(input)));
  });
  await fs.writeFile(output, `${lines.join("\n")}\n`, {
    flag: "wx",
    mode: 0o600,
  });
}

export async function prepareDatasetSelection(
  source: string,
  destination: string,
  label = "Dataset",
  copySingleFile = true,
) {
  const selected = await assertExistingPath(source, label);
  const stat = await fs.stat(selected);
  if (stat.isDirectory()) return selected;
  if (
    !stat.isFile() ||
    ![".json", ".jsonl", ".ndjson", ".parquet"].includes(
      path.extname(selected).toLowerCase(),
    )
  )
    throw new Error(
      `${label} must be a folder or a JSON, JSONL, NDJSON, or Parquet file`,
    );
  if (path.extname(selected).toLowerCase() === ".parquet") return selected;
  if (!copySingleFile) return selected;
  await fs.mkdir(destination, { recursive: false, mode: 0o700 });
  const output = path.join(destination, "train.jsonl");
  if (path.extname(selected).toLowerCase() === ".json") {
    await writeJsonDocument(selected, output);
  } else await writeJsonLines(selected, output);
  return destination;
}

export async function buildOsAiArgs(
  input: TrainingRequest,
  sessionsRoot: string,
  modelsRoot = defaultModelsRoot,
) {
  if (!["auto", "mlx", "llama.cpp"].includes(input.engine))
    throw new Error("Invalid training engine");
  if (
    !["auto", "metal", "mps", "cuda", "vulkan", "cpu"].includes(
      input.accelerator,
    )
  )
    throw new Error("Invalid accelerator");
  if (!["fine-tuning", "alignment", "fine-tune-align"].includes(input.stage))
    throw new Error("Invalid training stage");
  if (!["auto", "sgd", "adamw"].includes(input.optimizer))
    throw new Error("Invalid optimizer");
  if (!["auto", "on", "off"].includes(input.multiGpu))
    throw new Error("Invalid multi-GPU setting");
  if (!alignmentTypes.has(input.alignmentType))
    throw new Error("Invalid alignment method");
  if (!splitModes.has(input.splitMode ?? "auto"))
    throw new Error("Invalid GGUF split mode");

  const args = ["train"];
  if (input.modelSource === "official") {
    const modelVersion = input.modelVersion ?? "v1";
    if (!["v1", "v2"].includes(modelVersion))
      throw new Error("Invalid osCode model version");
    if (
      !["xsmall", "small", "medium", "large"].includes(input.tier) ||
      (modelVersion === "v1" && input.tier === "xsmall")
    )
      throw new Error("Invalid official model tier");
    args.push(
      "--tier",
      input.tier,
      "--model-version",
      modelVersion,
      "--bundled-root",
      modelsRoot,
    );
  } else if (input.modelSource === "custom") {
    const folder = await assertDirectory(
      input.customModelFolder,
      "Custom model",
    );
    args.push(
      "--custom",
      path.basename(folder),
      "--custom-root",
      path.dirname(folder),
    );
  } else {
    throw new Error("Invalid model source");
  }

  args.push(
    "--engine",
    input.engine,
    "--accelerator",
    input.accelerator,
    "--stage",
    input.stage,
    "--optimizer",
    input.optimizer,
    "--multi-gpu",
    input.multiGpu,
    "--sessions-root",
    sessionsRoot,
    input.autoSettings ? "--auto-settings" : "--no-auto-settings",
  );

  const needsFineTune = input.stage !== "alignment";
  const needsAlignment = input.stage !== "fine-tuning";
  if (needsFineTune)
    args.push(
      input.fullContentContext
        ? "--full-content-context"
        : "--no-full-content-context",
    );
  if (!input.autoSettings || input.calibrationApplied) {
    pushOptional(
      args,
      "--batch-size",
      optionalInteger(input.batchSize, "Batch size", 1, 65_536),
    );
    if (!input.fullContentContext || input.calibrationApplied)
      pushOptional(
        args,
        "--max-seq-length",
        optionalInteger(
          input.maxSeqLength,
          "Max sequence length",
          32,
          1_048_576,
        ),
      );
    pushOptional(
      args,
      "--gguf-batch-size",
      optionalInteger(input.ggufBatchSize, "GGUF microbatch", 1, 65_536),
    );
    pushOptional(
      args,
      "--gguf-threads",
      optionalInteger(input.ggufThreads, "GGUF CPU threads", 1, 4_096),
    );
    if (needsFineTune) {
      pushOptional(
        args,
        "--rank",
        optionalInteger(input.rank, "LoRA rank", 1, 65_536),
      );
      pushOptional(
        args,
        "--num-layers",
        optionalInteger(input.numLayers, "Layers to adapt", 1, 65_536),
      );
      for (const target of input.targetModules ?? []) {
        if (!targetModules.has(target))
          throw new Error("Invalid target projection");
        args.push("--target-module", target);
      }
    }
  }

  pushOptional(
    args,
    "--distributed-workers",
    optionalInteger(
      input.distributedWorkers,
      "MLX distributed workers",
      0,
      4_096,
    ),
  );
  pushOptional(
    args,
    "--main-gpu",
    optionalInteger(input.mainGpu, "Main GPU index", 0, 4_096),
  );
  const splitMode = input.splitMode ?? "auto";
  if (splitMode !== "auto") args.push("--split-mode", splitMode);
  const split = tensorSplit(input.tensorSplit);
  if (split) args.push("--tensor-split", split);
  for (const device of deviceList(input.devices)) args.push("--device", device);

  if (needsFineTune) {
    args.push(
      "--data",
      await assertExistingPath(input.fineTuneData, "Fine-tuning dataset"),
    );
    args.push(
      "--epochs",
      String(positiveInteger(input.iterations, "Fine-tune epochs")),
    );
    pushOptional(
      args,
      "--learning-rate",
      optionalNumber(
        input.learningRate,
        "Fine-tune learning rate",
        0,
        Number.MAX_VALUE,
        false,
      ),
    );
    pushOptional(
      args,
      "--scale",
      optionalNumber(input.scale, "LoRA scale", 0, Number.MAX_VALUE, false),
    );
    pushOptional(
      args,
      "--dropout",
      optionalNumber(input.dropout, "Dropout", 0, 0.999999),
    );
    const imageWidth = optionalInteger(
      input.imageWidth,
      "Image width",
      16,
      65_536,
    );
    const imageHeight = optionalInteger(
      input.imageHeight,
      "Image height",
      16,
      65_536,
    );
    if ((imageWidth === null) !== (imageHeight === null))
      throw new Error("Image width and height must be set together");
    if (imageWidth !== null && imageHeight !== null)
      args.push("--image-size", String(imageWidth), String(imageHeight));
    args.push(
      "--video-fps",
      String(
        optionalNumber(
          input.videoFps ?? 2,
          "Video frames per second",
          0,
          240,
          false,
        ),
      ),
      "--video-max-frames",
      String(
        positiveInteger(
          input.videoMaxFrames ?? 32,
          "Maximum video frames",
          32_768,
        ),
      ),
    );
    pushOptional(
      args,
      "--assistant-token-id",
      optionalInteger(input.assistantTokenId, "Assistant token ID", 0),
    );
    pushOptional(
      args,
      "--seed",
      optionalInteger(input.seed, "Training seed", 0, 2_147_483_647),
    );
    pushOptional(
      args,
      "--gradient-accumulation-steps",
      optionalInteger(
        input.gradientAccumulationSteps,
        "Gradient accumulation",
        1,
        1_000_000,
      ),
    );
    if (typeof input.gradientCheckpoint === "boolean")
      args.push(
        input.gradientCheckpoint
          ? "--gradient-checkpointing"
          : "--no-gradient-checkpointing",
      );
    pushOptional(
      args,
      "--save-every",
      optionalInteger(input.saveEvery, "Save interval"),
    );
    pushOptional(
      args,
      "--steps-per-report",
      optionalInteger(input.stepsPerReport, "Report interval"),
    );
    pushOptional(
      args,
      "--steps-per-eval",
      optionalInteger(input.stepsPerEval, "Evaluation interval"),
    );
    pushOptional(
      args,
      "--val-batches",
      optionalInteger(input.validationBatches, "Validation batches", -1),
    );
    if (typeof input.maskPrompt === "boolean")
      args.push(input.maskPrompt ? "--mask-prompt" : "--no-mask-prompt");
  }
  if (needsAlignment) {
    args.push(
      "--alignment-data",
      await assertExistingPath(input.alignmentData, "Alignment dataset"),
      "--alignment-type",
      input.alignmentType,
      "--alignment-iterations",
      String(
        positiveInteger(input.alignmentIterations, "Alignment iterations"),
      ),
      input.liveRollouts ? "--live-rollouts" : "--no-live-rollouts",
      "--rollout-max-tokens",
      String(
        positiveInteger(input.rolloutMaxTokens, "Rollout token limit", 32_768),
      ),
      "--rollouts-per-prompt",
      String(
        positiveInteger(input.rolloutsPerPrompt, "Rollouts per prompt", 128),
      ),
    );
    pushOptional(
      args,
      "--alignment-learning-rate",
      optionalNumber(
        input.alignmentLearningRate,
        "Alignment learning rate",
        0,
        Number.MAX_VALUE,
        false,
      ),
    );
    pushOptional(
      args,
      "--alignment-beta",
      optionalNumber(
        input.alignmentBeta,
        "Alignment beta",
        0,
        Number.MAX_VALUE,
        false,
      ),
    );
    pushOptional(
      args,
      "--alignment-gamma",
      optionalNumber(input.alignmentGamma, "Alignment gamma", 0),
    );
    pushOptional(
      args,
      "--ppo-clip",
      optionalNumber(input.ppoClip, "PPO clip", 0, Number.MAX_VALUE, false),
    );
    pushOptional(
      args,
      "--rollout-temperature",
      optionalNumber(input.rolloutTemperature, "Rollout temperature", 0),
    );
    pushOptional(
      args,
      "--rollout-top-p",
      optionalNumber(input.rolloutTopP, "Rollout top-p", 0, 1, false),
    );
    pushOptional(
      args,
      "--rollout-seed",
      optionalInteger(input.rolloutSeed, "Rollout seed", 0, 2_147_483_647),
    );
  }
  if (input.stage === "alignment")
    args.push("--adapter", await assertExistingPath(input.adapter, "Adapter"));
  const sessionName = cleanName(
    input.sessionName ||
      `${input.modelSource === "custom" ? path.basename(input.customModelFolder || "custom") : input.tier}-${input.stage}`,
  );
  if (input.sessionName.trim()) args.push("--session-name", sessionName);
  return { args, sessionName };
}

function displayCommand(executable: string, args: string[]) {
  const quote = (value: string) =>
    /^[A-Za-z0-9_./:-]+$/.test(value)
      ? value
      : `"${value.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
  return [executable, ...args].map(quote).join(" ");
}

function running(pid?: number) {
  if (!pid || !Number.isSafeInteger(pid) || pid < 1) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function readState(file: string) {
  try {
    return JSON.parse(await fs.readFile(file, "utf8")) as SessionState;
  } catch {
    return null;
  }
}

function argumentValue(args: string[], flag: string) {
  const index = args.lastIndexOf(flag);
  return index >= 0 && index + 1 < args.length ? args[index + 1] : undefined;
}

function argumentValues(args: string[], flag: string) {
  return args.filter((_value, index) => args[index - 1] === flag);
}

function numericArgument(args: string[], flag: string) {
  const value = Number(argumentValue(args, flag));
  return Number.isFinite(value) ? value : undefined;
}

export function restoreLegacyRequest(
  job: WorkerJob,
  state: SessionState,
): Partial<TrainingRequest> {
  const args = job.args;
  const custom = argumentValue(args, "--custom");
  const customRoot = argumentValue(args, "--custom-root");
  const data = argumentValue(args, "--data") || "";
  const alignmentData = argumentValue(args, "--alignment-data") || "";
  const reusedDataset =
    Boolean(data && alignmentData) &&
    (data === alignmentData || data.includes(`${path.sep}.shared-fine-tuning`));
  return {
    sessionsRoot: path.dirname(state.sessionDirectory),
    modelSource: custom ? "custom" : "official",
    modelVersion: (argumentValue(args, "--model-version") ||
      "v1") as TrainingRequest["modelVersion"],
    tier: (argumentValue(args, "--tier") || "small") as TrainingRequest["tier"],
    customModelFolder:
      custom && customRoot ? path.join(customRoot, custom) : "",
    engine: (argumentValue(args, "--engine") ||
      "auto") as TrainingRequest["engine"],
    accelerator: (argumentValue(args, "--accelerator") ||
      "auto") as TrainingRequest["accelerator"],
    stage: (argumentValue(args, "--stage") ||
      job.stage) as TrainingRequest["stage"],
    fineTuneData: reusedDataset ? alignmentData : data,
    alignmentData,
    reuseDataset: reusedDataset,
    adapter: argumentValue(args, "--adapter") || "",
    alignmentType: (argumentValue(args, "--alignment-type") ||
      "auto") as TrainingRequest["alignmentType"],
    optimizer: (argumentValue(args, "--optimizer") ||
      "auto") as TrainingRequest["optimizer"],
    autoSettings: args.includes("--auto-settings"),
    fullContentContext: args.includes("--full-content-context"),
    multiGpu: (argumentValue(args, "--multi-gpu") ||
      "auto") as TrainingRequest["multiGpu"],
    liveRollouts: !args.includes("--no-live-rollouts"),
    sessionName: argumentValue(args, "--session-name") || state.name,
    iterations:
      numericArgument(args, "--epochs") ||
      numericArgument(args, "--iterations") ||
      job.iterations,
    alignmentIterations:
      numericArgument(args, "--alignment-iterations") ||
      job.alignmentIterations,
    batchSize: numericArgument(args, "--batch-size"),
    gradientAccumulationSteps: numericArgument(
      args,
      "--gradient-accumulation-steps",
    ),
    gradientCheckpoint: !args.includes("--no-gradient-checkpointing"),
    maxSeqLength: numericArgument(args, "--max-seq-length"),
    learningRate: numericArgument(args, "--learning-rate"),
    alignmentLearningRate: numericArgument(args, "--alignment-learning-rate"),
    rank: numericArgument(args, "--rank"),
    scale: numericArgument(args, "--scale"),
    numLayers: numericArgument(args, "--num-layers"),
    dropout: numericArgument(args, "--dropout"),
    imageWidth: numericArgument(args, "--image-size"),
    imageHeight: (() => {
      const index = args.lastIndexOf("--image-size");
      const value = Number(index >= 0 ? args[index + 2] : undefined);
      return Number.isFinite(value) ? value : undefined;
    })(),
    videoFps: numericArgument(args, "--video-fps") || 2,
    videoMaxFrames: numericArgument(args, "--video-max-frames") || 32,
    assistantTokenId: numericArgument(args, "--assistant-token-id"),
    seed: numericArgument(args, "--seed"),
    saveEvery: numericArgument(args, "--save-every"),
    stepsPerReport: numericArgument(args, "--steps-per-report"),
    stepsPerEval: numericArgument(args, "--steps-per-eval"),
    validationBatches: numericArgument(args, "--val-batches"),
    maskPrompt: !args.includes("--no-mask-prompt"),
    targetModules: argumentValues(
      args,
      "--target-module",
    ) as TrainingRequest["targetModules"],
    alignmentBeta: numericArgument(args, "--alignment-beta"),
    alignmentGamma: numericArgument(args, "--alignment-gamma"),
    ppoClip: numericArgument(args, "--ppo-clip"),
    rolloutMaxTokens: numericArgument(args, "--rollout-max-tokens"),
    rolloutsPerPrompt: numericArgument(args, "--rollouts-per-prompt"),
    rolloutTemperature: numericArgument(args, "--rollout-temperature"),
    rolloutTopP: numericArgument(args, "--rollout-top-p"),
    rolloutSeed: numericArgument(args, "--rollout-seed"),
    ggufBatchSize: numericArgument(args, "--gguf-batch-size"),
    ggufThreads: numericArgument(args, "--gguf-threads"),
    distributedWorkers: numericArgument(args, "--distributed-workers"),
    splitMode: (argumentValue(args, "--split-mode") ||
      "auto") as TrainingRequest["splitMode"],
    tensorSplit: argumentValue(args, "--tensor-split") || "",
    mainGpu: numericArgument(args, "--main-gpu"),
    devices: argumentValues(args, "--device").join(", "),
  };
}

async function readSession(file: string) {
  const state = await readState(file);
  if (!state) return state;
  const sessionDirectory = path.dirname(file);
  // The directory being scanned is authoritative. Never trust editable paths
  // inside state.json for file operations such as reveal or delete.
  state.sessionDirectory = sessionDirectory;
  state.logPath = path.join(sessionDirectory, "training.log");
  if (state.request) return state;
  try {
    const job = JSON.parse(
      await fs.readFile(path.join(path.dirname(file), "job.json"), "utf8"),
    ) as WorkerJob;
    if (job.schemaVersion === 1 && job.id === state.id)
      state.request = restoreLegacyRequest(job, state);
  } catch {
    // Older or externally created sessions can still be displayed without a form.
  }
  return state;
}

async function writePrivateJson(file: string, value: unknown) {
  await fs.writeFile(file, `${JSON.stringify(value, null, 2)}\n`, {
    mode: 0o600,
  });
}

async function waitForSessionStatus(
  file: string,
  expected: SessionState["status"],
  timeout = 8_000,
) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    const state = await readState(file);
    if (state?.status === expected) return state;
    if (expected === "pausing" && state?.status === "paused") return state;
    if (state && !activeStatuses.has(state.status)) return state;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(
    `Training did not ${expected === "paused" ? "pause" : "resume"}`,
  );
}

function sharedFineTuneRow(value: unknown, file: string, lineNumber: number) {
  if (!value || typeof value !== "object")
    throw new Error(`Expected a JSON object at ${file}:${lineNumber}`);
  const row = value as Record<string, unknown>;
  const present = (...keys: string[]) =>
    keys.some((key) => {
      const item = row[key];
      return (
        (typeof item === "string" && Boolean(item.trim())) ||
        (Array.isArray(item) && item.length > 0)
      );
    });
  const hasPrompt = present("prompt", "question", "query", "instruction");
  const hasPreference =
    hasPrompt &&
    present("chosen", "preferred", "accepted", "winner") &&
    present(
      "rejected",
      "non_preferred",
      "dispreferred",
      "unpreferred",
      "loser",
    );
  const hasRankedPair =
    hasPrompt &&
    present("response_j") &&
    present("response_k") &&
    "label" in row;
  const hasReward =
    hasPrompt &&
    present("response", "completion", "answer", "output") &&
    (Number.isFinite(Number(row.reward ?? row.score ?? row.value)) ||
      "label" in row);
  if (hasPreference || hasRankedPair || hasReward) return row;
  throw new Error(
    `Shared data needs a preference, binary-feedback, or scored-response row (${file}:${lineNumber})`,
  );
}

export async function prepareSharedFineTuneData(
  source: string,
  destination: string,
) {
  const sourceRoot = await assertDirectory(source, "Shared training dataset");
  await fs.mkdir(destination, { recursive: false, mode: 0o700 });
  let trainRows = 0;
  for (const split of ["train", "valid", "test"]) {
    const input = path.join(sourceRoot, `${split}.jsonl`);
    if (!(await fs.stat(input).catch(() => null))) continue;
    const output = path.join(destination, `${split}.jsonl`);
    const handle = await fs.open(output, "wx", 0o600);
    const lines = readline.createInterface({
      input: createReadStream(input, { encoding: "utf8" }),
      crlfDelay: Infinity,
    });
    let lineNumber = 0;
    let buffer = "";
    try {
      for await (const line of lines) {
        lineNumber += 1;
        if (!line.trim())
          throw new Error(`Blank dataset line at ${input}:${lineNumber}`);
        let parsed: unknown;
        try {
          parsed = JSON.parse(line);
        } catch {
          throw new Error(`Invalid JSON at ${input}:${lineNumber}`);
        }
        buffer += `${JSON.stringify(sharedFineTuneRow(parsed, input, lineNumber))}\n`;
        if (buffer.length >= 1024 * 1024) {
          await handle.write(buffer);
          buffer = "";
        }
        if (split === "train") trainRows += 1;
      }
      if (buffer) await handle.write(buffer);
    } finally {
      lines.close();
      await handle.close();
    }
  }
  if (trainRows < 1)
    throw new Error(
      "Shared training dataset is missing a non-empty train.jsonl",
    );
  return destination;
}

export class SessionService {
  private readonly metricImports = new Map<string, Promise<void>>();

  constructor(
    private readonly legacySessionsRoot: string,
    private readonly workerScript: string,
    private readonly preferences: () => Promise<Preferences>,
  ) {}

  async backendStatus(): Promise<BackendStatus> {
    let executable = "osai";
    try {
      executable = (await this.preferences()).backendExecutable || executable;
    } catch (error) {
      return {
        available: false,
        executable,
        version: "",
        message: `Could not read the saved osAi CLI setting: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
    return new Promise((resolve) => {
      let settled = false;
      let timer: NodeJS.Timeout | undefined;
      let output = "";
      const finish = (status: BackendStatus) => {
        if (settled) return;
        settled = true;
        if (timer) clearTimeout(timer);
        resolve(status);
      };
      const appendOutput = (chunk: Buffer) => {
        if (output.length >= BACKEND_STATUS_OUTPUT_LIMIT) return;
        output += chunk
          .toString("utf8")
          .slice(0, BACKEND_STATUS_OUTPUT_LIMIT - output.length);
      };
      let child;
      try {
        child = spawn(executable, ["--version"], {
          shell: false,
          windowsHide: true,
          stdio: ["ignore", "pipe", "pipe"],
        });
      } catch (error) {
        finish({
          available: false,
          executable,
          version: "",
          message: `osAi CLI could not start: ${error instanceof Error ? error.message : String(error)}`,
        });
        return;
      }
      timer = setTimeout(() => {
        finish({
          available: false,
          executable,
          version: "",
          message: "osAi CLI did not respond within 5 seconds",
        });
        try {
          child.kill("SIGKILL");
        } catch {
          // The status check has already completed; process cleanup is best effort.
        }
      }, BACKEND_STATUS_TIMEOUT_MS);
      timer.unref();
      child.stdout.on("data", appendOutput);
      child.stderr.on("data", appendOutput);
      child.once("error", (error) => {
        finish({
          available: false,
          executable,
          version: "",
          message:
            (error as NodeJS.ErrnoException).code === "ENOENT"
              ? "osAi CLI was not found"
              : `osAi CLI could not start: ${error.message}`,
        });
      });
      child.once("close", (code) => {
        const version = osAiVersionFromOutput(output);
        if (code === 0 && version) {
          finish({
            available: true,
            executable,
            version,
            message: version,
          });
          return;
        }
        finish({
          available: false,
          executable,
          version: "",
          message:
            code === 0
              ? "The selected executable is not the osAi CLI"
              : output.replace(/\s+/g, " ").trim().slice(-240) ||
                `osAi CLI exited with code ${code ?? "unknown"}`,
        });
      });
    });
  }

  async start(input: TrainingRequest) {
    const preferences = await this.preferences();
    const executable = preferences.backendExecutable || "osai";
    const sessionsRoot = await ensureSessionsRoot(
      input.sessionsRoot || preferences.sessionsRoot,
    );
    const id = randomUUID();
    const requestedName = cleanName(
      input.sessionName ||
        `${input.modelSource === "custom" ? path.basename(input.customModelFolder || "custom") : input.tier}-${input.stage}`,
    );
    const directory = path.join(
      sessionsRoot,
      `${timestamp()}-${requestedName}-${id.slice(0, 8)}`,
    );
    await fs.mkdir(sessionsRoot, { recursive: true, mode: 0o700 });
    await fs.mkdir(directory, { recursive: false, mode: 0o700 });
    let prepared = input;
    let command: Awaited<ReturnType<typeof buildOsAiArgs>>;
    try {
      prepared = { ...input };
      if (input.stage !== "alignment")
        prepared.fineTuneData = await prepareDatasetSelection(
          input.fineTuneData,
          path.join(directory, ".fine-tune-input"),
          "Fine-tuning dataset",
          false,
        );
      if (input.stage !== "fine-tuning")
        prepared.alignmentData =
          input.reuseDataset && input.stage === "fine-tune-align"
            ? prepared.fineTuneData
            : await prepareDatasetSelection(
                input.alignmentData,
                path.join(directory, ".alignment-input"),
                "Alignment dataset",
                false,
              );
      command = await buildOsAiArgs(prepared, directory);
    } catch (error) {
      await fs.rm(directory, { recursive: true, force: true });
      throw error;
    }
    const { args, sessionName } = command;
    const job: WorkerJob = {
      schemaVersion: 1,
      id,
      executable,
      args,
      sessionDirectory: directory,
      statePath: path.join(directory, "state.json"),
      logPath: path.join(directory, "training.log"),
      stopPath: path.join(directory, "stop.request"),
      pausePath: path.join(directory, "pause.request"),
      resumePath: path.join(directory, "resume.request"),
      checkpointRequestPath: path.join(directory, "checkpoint.request"),
      autoStopPath: path.join(directory, "auto-stop.setting"),
      stage: input.stage,
      iterations: input.iterations,
      alignmentIterations: input.alignmentIterations,
      createdAt: new Date().toISOString(),
    };
    const state: SessionState = {
      schemaVersion: 1,
      id,
      name: sessionName,
      status: "queued",
      phase: "preparing",
      progress: 0,
      indeterminate: true,
      autoStopEnabled: Boolean(input.autoStop),
      autoStopMessage: input.autoStop ? "Gathering a loss baseline" : undefined,
      message: "Starting the local osAi worker",
      createdAt: job.createdAt,
      sessionDirectory: directory,
      logPath: job.logPath,
      command: displayCommand(executable, args),
      request: {
        ...input,
        sessionsRoot,
        sessionName: input.sessionName || sessionName,
      },
    };
    const jobPath = path.join(directory, "job.json");
    await writePrivateJson(jobPath, job);
    await writePrivateJson(job.statePath, state);
    const workerLog = openSync(path.join(directory, "worker.log"), "a", 0o600);
    let worker;
    try {
      worker = spawn(process.execPath, [this.workerScript, jobPath], {
        cwd: directory,
        detached: true,
        stdio: ["ignore", workerLog, workerLog],
        windowsHide: true,
        env: { ...process.env, ELECTRON_RUN_AS_NODE: "1" },
      });
    } finally {
      closeSync(workerLog);
    }
    await new Promise<void>((resolve, reject) => {
      worker.once("spawn", resolve);
      worker.once("error", reject);
    }).catch(async (error) => {
      state.status = "failed";
      state.indeterminate = false;
      state.endedAt = new Date().toISOString();
      state.message = "The local training worker could not start";
      state.error = error instanceof Error ? error.message : String(error);
      await writePrivateJson(job.statePath, state);
      throw error;
    });
    worker.unref();
    return state;
  }

  async list() {
    const preferences = await this.preferences();
    const currentRoot = await ensureSessionsRoot(preferences.sessionsRoot);
    const roots = [
      ...new Set([
        currentRoot,
        ...preferences.sessionRoots
          .filter(path.isAbsolute)
          .map((root) => path.resolve(root)),
        path.resolve(this.legacySessionsRoot),
      ]),
    ];
    const entries = (
      await Promise.all(
        roots.map(async (root) => {
          const children = await fs
            .readdir(root, { withFileTypes: true })
            .catch(() => []);
          return children
            .filter((entry) => entry.isDirectory())
            .map((entry) => path.join(root, entry.name, "state.json"));
        }),
      )
    ).flat();
    const states = await Promise.all(entries.map(readSession));
    const valid = states.filter((state): state is SessionState =>
      Boolean(state?.id),
    );
    await Promise.all(
      valid.map(async (state) => {
        if (!activeStatuses.has(state.status) || running(state.workerPid))
          return;
        if (Date.now() - Date.parse(state.createdAt) < 15_000) return;
        state.status = "failed";
        state.indeterminate = false;
        state.endedAt = new Date().toISOString();
        state.message = "The detached training worker ended unexpectedly";
        const workerLogPath = path.join(state.sessionDirectory, "worker.log");
        const workerOutput = await fs
          .readFile(workerLogPath, "utf8")
          .catch(() => "");
        const lastDiagnostic = workerOutput
          .trim()
          .split(/\r?\n/)
          .filter(Boolean)
          .at(-1);
        state.error ||= lastDiagnostic
          ? `Worker error: ${lastDiagnostic.slice(0, 500)} (see ${workerLogPath})`
          : `The worker stopped without an error message; check ${workerLogPath} and system logs`;
        await writePrivateJson(
          path.join(state.sessionDirectory, "state.json"),
          state,
        );
      }),
    );
    const visible = await Promise.all(valid.map(recoverVisibleProgress));
    return visible.sort((left, right) =>
      right.createdAt.localeCompare(left.createdAt),
    );
  }

  async find(id: string) {
    if (!/^[0-9a-f-]{36}$/i.test(id))
      throw new Error("Invalid session identifier");
    const state = (await this.list()).find((item) => item.id === id);
    if (!state) throw new Error("Training session not found");
    return state;
  }

  async remove(id: string, moveToTrash: (directory: string) => Promise<void>) {
    const state = await this.find(id);
    if (activeStatuses.has(state.status))
      throw new Error("Stop this training session before deleting it");
    const stateFile = path.join(state.sessionDirectory, "state.json");
    const stored = await readState(stateFile);
    if (!stored || stored.id !== id)
      throw new Error(
        "The training session changed before it could be deleted",
      );
    await moveToTrash(state.sessionDirectory);
  }

  async restart(
    id: string,
    input: TrainingRequest,
    moveToTrash: (directory: string) => Promise<void>,
  ) {
    const state = await this.find(id);
    if (activeStatuses.has(state.status))
      throw new Error("Stop this training session before restarting it");
    if (!state.request)
      throw new Error("This session does not contain restorable settings");

    const stateFile = path.join(state.sessionDirectory, "state.json");
    const stored = await readState(stateFile);
    if (!stored || stored.id !== id)
      throw new Error(
        "The training session changed before it could be restarted",
      );

    const sessionsRoot = await ensureSessionsRoot(
      input.sessionsRoot || path.dirname(state.sessionDirectory),
    );
    const request = { ...input, sessionsRoot };
    await buildOsAiArgs(request, sessionsRoot);
    if (request.stage !== "alignment")
      await assertExistingPath(request.fineTuneData, "Fine-tuning dataset");
    if (request.stage !== "fine-tuning")
      await assertExistingPath(request.alignmentData, "Alignment dataset");

    await moveToTrash(state.sessionDirectory);
    try {
      return await this.start(request);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(
        `The previous pipeline was moved to Trash, but the restart could not begin: ${detail}`,
      );
    }
  }

  async stop(id: string) {
    const state = await this.find(id);
    if (!activeStatuses.has(state.status)) return state;
    await fs
      .writeFile(path.join(state.sessionDirectory, "stop.request"), "stop\n", {
        flag: "wx",
        mode: 0o600,
      })
      .catch((error: NodeJS.ErrnoException) => {
        if (error.code !== "EEXIST") throw error;
      });
    state.status = "stopping";
    state.message = "Stopping after the current backend operation";
    await writePrivateJson(
      path.join(state.sessionDirectory, "state.json"),
      state,
    );
    return state;
  }

  private async control(id: string, action: "pause" | "resume") {
    const state = await this.find(id);
    const required = action === "pause" ? "running" : "paused";
    if (state.status !== required)
      throw new Error(
        action === "pause"
          ? "Only a running session can be paused"
          : "Only a paused session can be resumed",
      );
    const job = JSON.parse(
      await fs.readFile(path.join(state.sessionDirectory, "job.json"), "utf8"),
    ) as WorkerJob;
    if (!job.pausePath || !job.resumePath || !job.checkpointRequestPath)
      throw new Error(
        "Pause is available for sessions started by this version of osAi",
      );
    const request = action === "pause" ? job.pausePath : job.resumePath;
    const opposite = action === "pause" ? job.resumePath : job.pausePath;
    await fs.rm(opposite, { force: true });
    await fs.writeFile(request, `${action}\n`, { mode: 0o600 });
    if (action === "pause")
      return waitForSessionStatus(
        path.join(state.sessionDirectory, "state.json"),
        "pausing",
      );
    return waitForSessionStatus(
      path.join(state.sessionDirectory, "state.json"),
      "running",
    );
  }

  async pause(id: string) {
    return this.control(id, "pause");
  }

  async resume(id: string) {
    return this.control(id, "resume");
  }

  async checkpoint(id: string) {
    const state = await this.find(id);
    if (state.status !== "running" && state.status !== "paused")
      throw new Error("Start or resume training before saving a checkpoint");
    const job = JSON.parse(
      await fs.readFile(path.join(state.sessionDirectory, "job.json"), "utf8"),
    ) as WorkerJob;
    const request = path.join(state.sessionDirectory, "checkpoint.request");
    if (
      !job.checkpointRequestPath ||
      path.resolve(job.checkpointRequestPath) !== request
    )
      throw new Error(
        "Checkpoint saving is available for sessions started by this version of osAi",
      );
    const pending = `${request}.${process.pid}.pending`;
    await fs.writeFile(pending, `${randomUUID()}\n`, { mode: 0o600 });
    await fs.rename(pending, request);
    return { ...state, checkpointStatus: "requested" as const };
  }

  async setAutoStop(id: string, enabled: boolean) {
    const state = await this.find(id);
    if (!["queued", "running", "pausing", "paused"].includes(state.status))
      throw new Error("Auto stop can change only while training is active");
    const job = JSON.parse(
      await fs.readFile(path.join(state.sessionDirectory, "job.json"), "utf8"),
    ) as WorkerJob;
    const request = path.join(state.sessionDirectory, "auto-stop.setting");
    if (!job.autoStopPath || path.resolve(job.autoStopPath) !== request)
      throw new Error("Auto stop is available for newly started sessions");
    const pending = `${request}.${process.pid}.pending`;
    await fs.writeFile(pending, enabled ? "on\n" : "off\n", { mode: 0o600 });
    await fs.rename(pending, request);
    const started = Date.now();
    while (Date.now() - started < 8_000) {
      const updated = await readState(
        path.join(state.sessionDirectory, "state.json"),
      );
      if (updated?.autoStopEnabled === enabled) return updated;
      if (updated && !activeStatuses.has(updated.status)) return updated;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error("The trainer did not apply the Auto stop change");
  }

  async log(id: string) {
    const state = await this.find(id);
    const handle = await fs.open(state.logPath, "r").catch(() => null);
    if (!handle) return "";
    try {
      const stat = await handle.stat();
      const length = Math.min(stat.size, 160 * 1024);
      const buffer = Buffer.alloc(length);
      await handle.read(buffer, 0, length, Math.max(0, stat.size - length));
      return formatTerminalOutput(buffer.toString("utf8"));
    } finally {
      await handle.close();
    }
  }

  async metrics(id: string): Promise<TrainingMetric[]> {
    const state = await this.find(id);
    const file = path.join(state.sessionDirectory, "metrics.csv");
    await this.importLegacyMetrics(state);
    if (!(await fs.stat(file).catch(() => null))) return [];
    const lines = readline.createInterface({
      input: createReadStream(file, { encoding: "utf8" }),
      crlfDelay: Infinity,
    });
    let metrics: TrainingMetric[] = [];
    for await (const line of lines) {
      const metric = parseMetricRow(line);
      if (!metric) continue;
      metrics.push(metric);
      if (metrics.length > 4_000)
        metrics = metrics.filter(
          (entry, index) => entry.event === "checkpoint" || index % 2 === 0,
        );
    }
    return metrics;
  }

  private async importLegacyMetrics(state: SessionState) {
    const directory = state.sessionDirectory;
    const file = path.join(directory, "metrics.csv");
    const cursorFile = path.join(directory, "metrics-import.json");
    if (
      (await fs.stat(file).catch(() => null)) &&
      !(await fs.stat(cursorFile).catch(() => null))
    )
      return; // New workers write their own compact metric stream.
    const prior = this.metricImports.get(directory);
    if (prior) return prior;
    const importTask = (async () => {
      const log = path.join(directory, "training.log");
      const stat = await fs.stat(log).catch(() => null);
      if (!stat?.size) return;
      type Cursor = {
        offset: number;
        steps: number;
        mlxTable: boolean;
        percent: number;
        lastStep?: number | null;
        last: Record<string, number>;
      };
      const cursor = await fs
        .readFile(cursorFile, "utf8")
        .then((raw) => JSON.parse(raw) as Cursor)
        .catch((): Cursor => ({
          offset: 0,
          steps: 0,
          mlxTable: false,
          percent: 0,
          lastStep: null,
          last: {},
        }));
      if (cursor.offset > stat.size) {
        cursor.offset = 0;
        cursor.last = {};
        cursor.lastStep = null;
        await fs.writeFile(file, metricHeader, { mode: 0o600 });
      } else if (!(await fs.stat(file).catch(() => null))) {
        await fs.writeFile(file, metricHeader, { mode: 0o600 });
      }
      if (cursor.offset === stat.size) return;
      const lines = readline.createInterface({
        input: createReadStream(log, {
          start: cursor.offset,
          end: stat.size - 1,
          encoding: "utf8",
        }),
        crlfDelay: Infinity,
      });
      let output = "";
      for await (const line of lines) {
        const steps = /\bosai: training plan\b.*?\bsteps=(\d+)/.exec(line);
        if (steps) cursor.steps = Number(steps[1]);
        if (/^iter\s+train_loss\s+tok\/s\s+tokens$/i.test(line.trim()))
          cursor.mlxTable = true;
        const data = /\bdata=(\d+)\/(\d+)/.exec(line);
        const mlx = cursor.mlxTable
          ? /^\s*(\d[\d,]*)\s+([\d.eE+-]+)\s/.exec(line)
          : null;
        if (data) cursor.lastStep = Number(data[1]);
        else if (mlx) cursor.lastStep = Number(mlx[1].replaceAll(",", ""));
        if (data && Number(data[2]) > 0)
          cursor.percent = Math.min(
            100,
            (100 * Number(data[1])) / Number(data[2]),
          );
        else if (mlx && cursor.steps > 0)
          cursor.percent = Math.min(
            100,
            (100 * Number(mlx[1].replaceAll(",", ""))) / cursor.steps,
          );
        const checkpoint = /osai: checkpoint saved\b/i.test(line);
        if (checkpoint) {
          output += metricRow({
            time: "",
            percent: cursor.percent,
            step: cursor.lastStep ?? null,
            device: "Trainer",
            loss: null,
            lossUncertainty: null,
            accuracy: null,
            accuracyUncertainty: null,
            event: "checkpoint",
          });
        } else if (data || mlx || /^\s*Iter\s+\d+/i.test(line)) {
          const metric = lossMetric(line, cursor.percent, cursor.mlxTable);
          if (metric) {
            const previous = cursor.last[metric.device];
            if (previous === undefined || metric.percent - previous >= 0.1) {
              cursor.last[metric.device] = metric.percent;
              output += metricRow({ ...metric, time: "" });
            }
          }
        }
        if (output.length >= 32 * 1024) {
          await fs.appendFile(file, output, "utf8");
          output = "";
        }
      }
      if (output) await fs.appendFile(file, output, "utf8");
      cursor.offset = stat.size;
      const pending = `${cursorFile}.${process.pid}.pending`;
      await fs.writeFile(pending, JSON.stringify(cursor), { mode: 0o600 });
      await fs.rename(pending, cursorFile);
    })().finally(() => this.metricImports.delete(directory));
    this.metricImports.set(directory, importTask);
    return importTask;
  }

  async root() {
    return ensureSessionsRoot((await this.preferences()).sessionsRoot);
  }
}
