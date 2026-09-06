import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import { createReadStream } from "node:fs";
import type {
  BackendStatus,
  Preferences,
  SessionState,
  TrainingRequest,
  WorkerJob,
} from "../types.js";

const activeStatuses = new Set(["queued", "running", "paused", "stopping"]);
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
const splitModes = new Set(["auto", "none", "layer", "row", "tensor"]);

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
      buffer += `${JSON.stringify(row)}\n`;
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
        : [parsed];
  if (records.length < 1) throw new Error(`Dataset file is empty: ${input}`);
  const lines = records.map((row) => {
    if (!row || typeof row !== "object" || Array.isArray(row))
      throw new Error(`Dataset rows must be JSON objects: ${input}`);
    return JSON.stringify(row);
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
) {
  const selected = await assertExistingPath(source, label);
  const stat = await fs.stat(selected);
  if (stat.isDirectory()) return selected;
  if (
    !stat.isFile() ||
    ![".json", ".jsonl"].includes(path.extname(selected).toLowerCase())
  )
    throw new Error(`${label} must be a folder or a .json/.jsonl file`);
  await fs.mkdir(destination, { recursive: false, mode: 0o700 });
  const output = path.join(destination, "train.jsonl");
  if (path.extname(selected).toLowerCase() === ".json") {
    await writeJsonDocument(selected, output);
  } else {
    await fs.link(selected, output).catch(async () => {
      await fs.copyFile(selected, output, fs.constants.COPYFILE_EXCL);
    });
  }
  return destination;
}

export async function buildOsAiArgs(
  input: TrainingRequest,
  sessionsRoot: string,
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
    if (!["small", "medium", "large"].includes(input.tier))
      throw new Error("Invalid official model tier");
    args.push("--tier", input.tier);
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
  if (!input.autoSettings) {
    pushOptional(
      args,
      "--batch-size",
      optionalInteger(input.batchSize, "Batch size", 1, 65_536),
    );
    pushOptional(
      args,
      "--max-seq-length",
      optionalInteger(input.maxSeqLength, "Max sequence length", 32, 1_048_576),
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
      await assertDirectory(input.fineTuneData, "Fine-tuning dataset"),
    );
    args.push(
      "--iterations",
      String(positiveInteger(input.iterations, "Iterations")),
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
      await assertDirectory(input.alignmentData, "Alignment dataset"),
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
    data.includes(`${path.sep}.shared-fine-tuning`) && Boolean(alignmentData);
  return {
    sessionsRoot: path.dirname(state.sessionDirectory),
    modelSource: custom ? "custom" : "official",
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
    multiGpu: (argumentValue(args, "--multi-gpu") ||
      "auto") as TrainingRequest["multiGpu"],
    liveRollouts: !args.includes("--no-live-rollouts"),
    sessionName: argumentValue(args, "--session-name") || state.name,
    iterations: numericArgument(args, "--iterations") || job.iterations,
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
  if (!state || state.request) return state;
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
  const prompt = typeof row.prompt === "string" ? row.prompt.trim() : "";
  const completion =
    typeof row.completion === "string" ? row.completion.trim() : "";
  const chosen = typeof row.chosen === "string" ? row.chosen.trim() : "";
  const rejected = typeof row.rejected === "string" ? row.rejected.trim() : "";
  const response = typeof row.response === "string" ? row.response.trim() : "";
  const reward = Number(row.reward);
  const hasPreference = Boolean(prompt && chosen && rejected);
  const hasReward = Boolean(prompt && response && Number.isFinite(reward));
  if (prompt && completion && (hasPreference || hasReward)) return row;
  if (hasPreference) return { prompt, completion: chosen };
  if (hasReward) return { prompt, completion: response };
  throw new Error(
    `Shared data needs prompt/chosen/rejected or prompt/response/reward rows (${file}:${lineNumber})`,
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
  constructor(
    private readonly legacySessionsRoot: string,
    private readonly workerScript: string,
    private readonly preferences: () => Promise<Preferences>,
  ) {}

  async backendStatus(): Promise<BackendStatus> {
    const executable = (await this.preferences()).backendExecutable || "osai";
    return new Promise((resolve) => {
      const child = spawn(executable, ["--version"], {
        shell: false,
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
      });
      let output = "";
      const timer = setTimeout(() => child.kill(), 8_000);
      timer.unref();
      child.stdout.on("data", (chunk: Buffer) => (output += chunk.toString()));
      child.stderr.on("data", (chunk: Buffer) => (output += chunk.toString()));
      child.once("error", (error) => {
        clearTimeout(timer);
        resolve({
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
        clearTimeout(timer);
        const version = output.trim().slice(0, 120);
        resolve({
          available: code === 0,
          executable,
          version: code === 0 ? version : "",
          message:
            code === 0
              ? version
              : output.trim().slice(-240) || `osAi exited with ${code}`,
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
      if (input.stage === "fine-tune-align" && input.reuseDataset) {
        const sharedSource = await prepareDatasetSelection(
          input.fineTuneData,
          path.join(directory, ".shared-input"),
          "Shared training dataset",
        );
        const shared = await prepareSharedFineTuneData(
          sharedSource,
          path.join(directory, ".shared-fine-tuning"),
        );
        prepared = {
          ...input,
          fineTuneData: shared,
          alignmentData: sharedSource,
        };
      } else {
        prepared = { ...input };
        if (input.stage !== "alignment")
          prepared.fineTuneData = await prepareDatasetSelection(
            input.fineTuneData,
            path.join(directory, ".fine-tune-input"),
            "Fine-tuning dataset",
          );
        if (input.stage !== "fine-tuning")
          prepared.alignmentData = await prepareDatasetSelection(
            input.alignmentData,
            path.join(directory, ".alignment-input"),
            "Alignment dataset",
          );
      }
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
    const worker = spawn(process.execPath, [this.workerScript, jobPath], {
      cwd: directory,
      detached: true,
      stdio: "ignore",
      windowsHide: true,
      env: { ...process.env, ELECTRON_RUN_AS_NODE: "1" },
    });
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
        state.error ||= "No live worker process was found";
        await writePrivateJson(
          path.join(state.sessionDirectory, "state.json"),
          state,
        );
      }),
    );
    return valid.sort((left, right) =>
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
    if (!job.pausePath || !job.resumePath)
      throw new Error(
        "Pause is available for sessions started by this version of osAi",
      );
    const request = action === "pause" ? job.pausePath : job.resumePath;
    const opposite = action === "pause" ? job.resumePath : job.pausePath;
    await fs.rm(opposite, { force: true });
    await fs.writeFile(request, `${action}\n`, { mode: 0o600 });
    return waitForSessionStatus(
      path.join(state.sessionDirectory, "state.json"),
      action === "pause" ? "paused" : "running",
    );
  }

  async pause(id: string) {
    return this.control(id, "pause");
  }

  async resume(id: string) {
    return this.control(id, "resume");
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
      return buffer.toString("utf8");
    } finally {
      await handle.close();
    }
  }

  async root() {
    return ensureSessionsRoot((await this.preferences()).sessionsRoot);
  }
}
