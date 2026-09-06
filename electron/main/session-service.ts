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

const activeStatuses = new Set(["queued", "running", "stopping"]);
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

async function writePrivateJson(file: string, value: unknown) {
  await fs.writeFile(file, `${JSON.stringify(value, null, 2)}\n`, {
    mode: 0o600,
  });
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
    private readonly sessionsRoot: string,
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
    const id = randomUUID();
    const requestedName = cleanName(
      input.sessionName ||
        `${input.modelSource === "custom" ? path.basename(input.customModelFolder || "custom") : input.tier}-${input.stage}`,
    );
    const directory = path.join(
      this.sessionsRoot,
      `${timestamp()}-${requestedName}-${id.slice(0, 8)}`,
    );
    await fs.mkdir(this.sessionsRoot, { recursive: true, mode: 0o700 });
    await fs.mkdir(directory, { recursive: false, mode: 0o700 });
    let prepared = input;
    let command: Awaited<ReturnType<typeof buildOsAiArgs>>;
    try {
      if (input.stage === "fine-tune-align" && input.reuseDataset) {
        const shared = await prepareSharedFineTuneData(
          input.fineTuneData,
          path.join(directory, ".shared-fine-tuning"),
        );
        prepared = {
          ...input,
          fineTuneData: shared,
          alignmentData: input.fineTuneData,
        };
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
    await fs.mkdir(this.sessionsRoot, { recursive: true, mode: 0o700 });
    const entries = await fs.readdir(this.sessionsRoot, {
      withFileTypes: true,
    });
    const states = await Promise.all(
      entries
        .filter((entry) => entry.isDirectory())
        .map((entry) =>
          readState(path.join(this.sessionsRoot, entry.name, "state.json")),
        ),
    );
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

  root() {
    return this.sessionsRoot;
  }
}
