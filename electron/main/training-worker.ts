import { spawn, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import { createWriteStream, writeFileSync } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { StringDecoder } from "node:string_decoder";
import type { SessionState, WorkerJob } from "../types.js";
import { backendRuntimeEnvironment } from "./backend-source.js";
import { LossRiseDetector } from "./auto-stop.js";
import {
  cleanTerminalLine,
  FineTuneProgressParser,
  phaseProgress,
} from "./training-progress.js";
import {
  lossMetric,
  metricHeader,
  metricRow,
  type TrainingMetric,
} from "./training-metrics.js";

const jobPath = process.argv[2];
if (!jobPath || !path.isAbsolute(jobPath)) process.exit(2);

let state: SessionState;
let job: WorkerJob;
let child: ChildProcess | null = null;
let stopping = false;
let forcedStop = false;
let legacyStop = false;
let stopTimer: ReturnType<typeof setTimeout> | null = null;
let paused = false;
let pauseGeneration = "";
let stopGeneration = "";
let autoStopGeneration = "";
let autoStopRequestedAt = 0;
let autoStopped = false;
let autoStoppedAtPriorCheckpoint = false;
let pausedIndeterminate = false;
let finalised = false;
let lastStderrLine = "";
let actionableStderrLine = "";
let actionableStderrScore = 0;
let writeTimer: ReturnType<typeof setTimeout> | null = null;
let writeChain = Promise.resolve();
let fineTuneProgress: FineTuneProgressParser;
let lastCheckpointRequest = "";
let metricWrites = Promise.resolve();
let mlxMetricTable = false;
let lastTrainingStep: number | null = null;
const lastMetric = new Map<string, { at: number; percent: number }>();
const lossRise = new LossRiseDetector();
const autoStopCheckpointTimeoutMs = 2 * 60_000;
const gracefulStopTimeoutMs = 10 * 60_000;

function supportsExactStop() {
  const engineIndex = job.args.indexOf("--engine");
  const engine = engineIndex >= 0 ? job.args[engineIndex + 1] : "auto";
  return (
    job.stage === "fine-tuning" &&
    (engine === "llama.cpp" ||
      (engine === "auto" && process.platform !== "darwin"))
  );
}

function disableAutoStop(message: string) {
  state.autoStopEnabled = false;
  state.autoStopMessage = message;
  if (state.request) state.request.autoStop = false;
  lossRise.setEnabled(false);
  if (job.autoStopPath) {
    try {
      writeFileSync(job.autoStopPath, "off\n", { mode: 0o600 });
    } catch (error) {
      state.error = error instanceof Error ? error.message : String(error);
    }
  }
}

function recordMetric(metric: TrainingMetric, force = false) {
  const previous = lastMetric.get(metric.device);
  const now = Date.now();
  if (
    !force &&
    previous &&
    now - previous.at < 15_000 &&
    metric.percent - previous.percent < 10
  )
    return;
  lastMetric.set(metric.device, { at: now, percent: metric.percent });
  const file = path.join(job.sessionDirectory, "metrics.csv");
  metricWrites = metricWrites
    .catch(() => undefined)
    .then(() => fs.appendFile(file, metricRow(metric), { encoding: "utf8" }));
  void metricWrites.catch((error) =>
    console.error("osAi metric write failed:", error),
  );
}

async function atomicStateWrite() {
  const snapshot = JSON.stringify(state, null, 2);
  writeChain = writeChain
    .catch(() => undefined)
    .then(async () => {
      const temporary = `${job.statePath}.${process.pid}.tmp`;
      for (let attempt = 0; attempt < 5; attempt += 1) {
        try {
          await fs.writeFile(temporary, `${snapshot}\n`, { mode: 0o600 });
          await fs.rename(temporary, job.statePath);
          return;
        } catch (error) {
          const code = (error as NodeJS.ErrnoException).code;
          if (
            !["EACCES", "EPERM", "EBUSY"].includes(code || "") ||
            attempt === 4
          )
            throw error;
          await new Promise((resolve) =>
            setTimeout(resolve, 100 * 2 ** attempt),
          );
        }
      }
    });
  await writeChain;
}

function scheduleStateWrite(immediate = false) {
  if (immediate) {
    if (writeTimer) clearTimeout(writeTimer);
    writeTimer = null;
    return atomicStateWrite();
  }
  // Throttle while output is streaming. Debouncing from the last line made a
  // continuously busy trainer appear stuck at its initial percentage.
  if (writeTimer) return Promise.resolve();
  writeTimer = setTimeout(() => {
    writeTimer = null;
    void atomicStateWrite().catch((error) => {
      console.error("osAi worker state write failed:", error);
    });
  }, 250);
  writeTimer.unref();
  return Promise.resolve();
}

function clamp(value: number) {
  return Math.max(state.progress, Math.min(99, Math.round(value)));
}

function consumeLine(raw: string) {
  const line = cleanTerminalLine(raw);
  if (!line) return;
  const checkpoint =
    /osai: checkpoint saved path=(.+?) generation=([^\s]+)/i.exec(line);
  if (checkpoint) {
    if (checkpoint[2] !== "auto" && checkpoint[2] !== "final")
      lastCheckpointRequest = checkpoint[2];
    const savedPath = path.resolve(checkpoint[1]);
    const relative = path.relative(job.sessionDirectory, savedPath);
    if (
      !relative.startsWith("..") &&
      !path.isAbsolute(relative) &&
      /[\\/](?:outputs|\.internal)[\\/]checkpoint[\\/]/i.test(savedPath)
    ) {
      state.checkpointPath = savedPath;
      state.checkpointStatus = "requested";
      state.message = "Adapter saved; merging model weights";
      void scheduleStateWrite(true);
    }
    return;
  }
  const checkpointModel =
    /osai: checkpoint model ready path=(.+?)(?: generation=([^\s]+))?$/i.exec(
      line,
    );
  if (checkpointModel) {
    const modelPath = path.resolve(checkpointModel[1]);
    const merged = path.join(modelPath, "merged.gguf");
    const adapter = path.join(modelPath, "osai_adapter.gguf");
    const mlxMerged = path.join(
      modelPath,
      "merged",
      "model.safetensors.index.json",
    );
    const mlxAdapter = path.join(
      modelPath,
      "osai_adapter",
      "adapters.safetensors",
    );
    void Promise.all([
      fs.stat(merged).catch(() => null),
      fs.stat(adapter).catch(() => null),
      fs.stat(mlxMerged).catch(() => null),
      fs.stat(mlxAdapter).catch(() => null),
    ]).then(([ggufModel, ggufAdapter, mlxModel, mlxAdapterFile]) => {
      if (
        (!ggufModel?.size || !ggufAdapter?.size) &&
        (!mlxModel?.size || !mlxAdapterFile?.size)
      ) {
        state.checkpointStatus = "failed";
        state.error = `Checkpoint output is missing an adapter or merged model: ${modelPath}`;
        void scheduleStateWrite(true);
        return;
      }
      state.checkpointModelPath = modelPath;
      state.checkpointStatus = "saved";
      state.checkpointSavedAt = new Date().toISOString();
      recordMetric(
        {
          time: state.checkpointSavedAt,
          percent: state.trainingPercent ?? 0,
          step: lastTrainingStep,
          device: "Trainer",
          loss: null,
          lossUncertainty: null,
          accuracy: null,
          accuracyUncertainty: null,
          event: "checkpoint",
        },
        true,
      );
      void scheduleStateWrite(true);
      if (state.checkpointPath && checkpointModel[2] === pauseGeneration)
        void completePause(state.checkpointPath);
      if (state.checkpointPath && checkpointModel[2] === autoStopGeneration)
        void completeAutomaticStop(state.checkpointPath);
      if (checkpointModel[2] === stopGeneration) void completeStopCheckpoint();
    });
    return;
  }
  if (/osai: checkpoint model failed\b/i.test(line)) {
    state.checkpointStatus = "failed";
    state.error = line.slice(0, 1000);
    void scheduleStateWrite(true);
    return;
  }
  if (/osai: checkpoint failed\b/i.test(line)) {
    state.checkpointStatus = "failed";
    if (autoStopGeneration) {
      autoStopGeneration = "";
      state.status = "running";
      disableAutoStop(
        "Auto stop could not save a checkpoint; training continues",
      );
    }
    if (pauseGeneration) {
      pauseGeneration = "";
      state.status = "running";
      state.message = "Pause could not save a checkpoint; training continues";
    }
    void scheduleStateWrite(true);
    return;
  }
  if (/^osai: full content scan\b/i.test(line)) {
    const records = /\brecords=(\d+)/.exec(line)?.[1];
    state.phase = "preparing";
    state.indeterminate = true;
    state.trainingPercent = undefined;
    state.message = records
      ? `Scanning ${Number(records).toLocaleString()} records for maximum context`
      : "Scanning the selected dataset for maximum context";
    void scheduleStateWrite();
    return;
  }
  if (/^osai: preparing dataset\b/i.test(line)) {
    const split = /\bsplit=(\w+)/.exec(line)?.[1] || "training";
    state.phase = "preparing";
    state.indeterminate = true;
    state.trainingPercent = undefined;
    state.message = `Converting and merging ${split} dataset files locally`;
    void scheduleStateWrite();
    return;
  }
  if (/^osai: full content\b/i.test(line)) {
    const context = /\bcontext=(\d+)/.exec(line)?.[1];
    const records = /\brecords=(\d+)/.exec(line)?.[1];
    state.phase = "preparing";
    state.indeterminate = true;
    state.message = context
      ? `Largest of ${Number(records || 0).toLocaleString()} records selected · ${Number(context).toLocaleString()} token context`
      : "Largest training record selected as context";
    if (state.autoSettingsSummary && context)
      state.autoSettingsSummary = state.autoSettingsSummary.replace(
        /Context \d+/,
        `Context ${context}`,
      );
    void scheduleStateWrite();
    return;
  }
  if (/^osai: auto settings\b/i.test(line)) {
    const fields = Object.fromEntries(
      [...line.matchAll(/\b([a-z_]+)=([^\s]+)/gi)].map((match) => [
        match[1],
        match[2],
      ]),
    );
    state.autoSettingsSummary = [
      fields.profile && `Profile ${fields.profile}`,
      fields.context && `Context ${fields.context}`,
      fields.batch && `Batch ${fields.batch}`,
      fields.layers && `Layers ${fields.layers}`,
      fields.rank && `Rank ${fields.rank}`,
      fields.gguf_batch && `GGUF microbatch ${fields.gguf_batch}`,
      fields.threads && `Threads ${fields.threads}`,
      fields.budget && `Memory budget ${fields.budget}`,
    ]
      .filter(Boolean)
      .join(" · ");
    state.message = "Auto settings selected for this hardware";
    void scheduleStateWrite();
    return;
  }
  if (/^osai: training plan\b/i.test(line)) {
    fineTuneProgress.consume(line);
    const windows = /\bwindows=(\d+)/.exec(line)?.[1];
    state.phase = "fine-tuning";
    state.indeterminate = true;
    state.trainingPercent = undefined;
    state.message = windows
      ? `Preparing ${Number(windows).toLocaleString()} training windows`
      : "Preparing training windows and optimizer steps";
    void scheduleStateWrite();
    return;
  }
  const retry =
    /^osai: auto retry\s+.*?\bengine=([^\s]+).*?\battempt=(\d+).*?\bcontext=(\d+)->(\d+).*?\bbatch=(\d+)->(\d+)/i.exec(
      line,
    );
  if (retry) {
    const [, engine, attempt, previous, next, oldBatch, newBatch] = retry;
    state.phase = "fine-tuning";
    state.indeterminate = true;
    state.trainingPercent = undefined;
    state.adjustment = `Attempt ${attempt} · ${engine} · context ${previous} → ${next} · batch ${oldBatch} → ${newBatch}`;
    if (state.autoSettingsSummary) {
      state.autoSettingsSummary = state.autoSettingsSummary
        .replace(/Context \d+/, `Context ${next}`)
        .replace(/Batch \d+/, `Batch ${newBatch}`);
    }
    state.message = "Memory limit reached; preparing a smaller training window";
    fineTuneProgress = new FineTuneProgressParser(job.iterations);
    void scheduleStateWrite();
    return;
  }
  const attempt =
    /^osai: training attempt\s+.*?\bengine=([^\s]+).*?\battempt=(\d+).*?\bcontext=(\d+).*?\bbatch=(\d+)/i.exec(
      line,
    );
  if (attempt) {
    state.phase = "fine-tuning";
    state.indeterminate = true;
    state.trainingPercent = undefined;
    state.message = `Preparing ${attempt[1]} training attempt ${attempt[2]} with ${attempt[3]} token windows`;
    void scheduleStateWrite();
    return;
  }
  const lower = line.toLowerCase();
  if (/^iter\s+train_loss\s+tok\/s\s+tokens$/i.test(line))
    mlxMetricTable = true;
  const percent = /(?:^|\s)(\d{1,3})%(?:\s|$)/.exec(line);
  const byteProgress =
    /\b\d+(?:\.\d+)?(?:B|KiB|MiB|GiB)\s*\/\s*\d+(?:\.\d+)?(?:B|KiB|MiB|GiB)\b/i.exec(
      line,
    );
  if ((lower.includes("download") || byteProgress) && percent) {
    state.phase = "download";
    state.indeterminate = false;
    state.progress = clamp(2 + Math.min(100, Number(percent[1])) * 0.03);
    const item = line
      .replace(/^.*?\b\d{1,3}%\s+\S+\s*/, "")
      .trim()
      .slice(0, 180);
    state.message = item
      ? item.toLowerCase().includes("catalogue")
        ? "Checking the model catalogue"
        : `Downloading ${item}${byteProgress ? ` · ${byteProgress[0].replace(/\s/g, "")}` : ""}`
      : "Downloading model files";
  } else if (
    lower.includes("rollout_source=") ||
    lower.includes("live rollout")
  ) {
    state.phase = "rollouts";
    state.indeterminate = true;
    state.progress = clamp(job.stage === "fine-tune-align" ? 66 : 10);
    state.message = "Generating and scoring fresh answers locally";
  } else {
    const alignment = /alignment_step\s*=\s*(\d+)/i.exec(line);
    const training = fineTuneProgress.consume(line);
    if (alignment) {
      const completed = Number(alignment[1]);
      state.phase = "alignment";
      state.indeterminate = false;
      state.trainingPercent = Math.min(
        100,
        (100 * completed) / Math.max(1, job.alignmentIterations),
      );
      state.progress = clamp(
        phaseProgress(
          completed,
          job.alignmentIterations,
          "alignment",
          job.stage,
        ),
      );
      state.message = `Alignment update ${completed} of ${job.alignmentIterations}`;
    } else if (training && state.phase !== "alignment") {
      const { completed, total } = training;
      state.phase = "fine-tuning";
      state.indeterminate = false;
      state.trainingPercent = Math.min(
        100,
        (100 * completed) / Math.max(1, total),
      );
      state.progress = clamp(
        phaseProgress(completed, total, "fine-tuning", job.stage),
      );
      const percentComplete = ((100 * completed) / Math.max(1, total)).toFixed(
        2,
      );
      if (!autoStopGeneration)
        state.message =
          `Fine-tuning: ${completed.toLocaleString()} of ${total.toLocaleString()} ` +
          `(${percentComplete}%)`;
    } else if (lower.includes("publish") || lower.includes("fusion")) {
      state.phase = "publishing";
      state.indeterminate = true;
      state.trainingPercent = undefined;
      state.progress = clamp(98);
      state.message = "Publishing the adapter and final model outputs";
    }
  }
  if (state.phase === "fine-tuning" && state.trainingPercent !== undefined) {
    const metric = lossMetric(line, state.trainingPercent, mlxMetricTable);
    if (metric) {
      if (metric.step !== null && metric.step !== undefined)
        lastTrainingStep = Math.max(lastTrainingStep ?? 0, metric.step);
      recordMetric(metric);
      if (state.status === "running") {
        const rise = lossRise.observe(metric);
        if (
          state.autoStopEnabled &&
          state.autoStopMessage === "Gathering a loss baseline" &&
          lossRise.ready()
        )
          state.autoStopMessage = "Watching average loss for a sustained rise";
        if (rise) void requestAutomaticStop(rise);
      }
    }
  }
  void scheduleStateWrite();
}

function consumeStderrLine(raw: string) {
  consumeLine(raw);
  const line = cleanTerminalLine(raw);
  if (!line) return;
  lastStderrLine = line.replace(/^osai:\s*/i, "").slice(0, 1000);
  const lower = lastStderrLine.toLowerCase();
  const score =
    /\b(missing|unsupported|invalid|failed|failure|error|cannot|could not|exceeds|required|locked)\b/.test(
      lower,
    ) && !/^command exited with status\b/.test(lower)
      ? 2
      : /^command exited with status\b/.test(lower)
        ? 0
        : 1;
  if (score >= actionableStderrScore) {
    actionableStderrLine = lastStderrLine;
    actionableStderrScore = score;
  }
}

function lineReader(consume: (line: string) => void) {
  let pending = "";
  const decoder = new StringDecoder("utf8");
  return (chunk: Buffer) => {
    pending += decoder.write(chunk);
    const lines = pending.split(/\r\n|\r|\n/);
    pending = lines.pop() || "";
    for (const line of lines) consume(line);
  };
}

async function runWindowsProcessControl(command: "Suspend" | "Resume") {
  if (!child?.pid) throw new Error("The training process is not running");
  const nativeMethod =
    command === "Suspend" ? "NtSuspendProcess" : "NtResumeProcess";
  const script = [
    'Add-Type -TypeDefinition \'using System; using System.Runtime.InteropServices; public static class OsAiNativeProcessControl { [DllImport("ntdll.dll")] public static extern uint NtSuspendProcess(IntPtr handle); [DllImport("ntdll.dll")] public static extern uint NtResumeProcess(IntPtr handle); }\'',
    `$rootPid = ${child.pid}`,
    "$inventory = @(Get-CimInstance Win32_Process -Property ProcessId,ParentProcessId -ErrorAction Stop)",
    "$targets = [System.Collections.Generic.List[int]]::new()",
    "$targets.Add($rootPid)",
    "for ($i = 0; $i -lt $targets.Count; $i++) { foreach ($item in $inventory) { if ($item.ParentProcessId -eq $targets[$i] -and -not $targets.Contains([int]$item.ProcessId)) { $targets.Add([int]$item.ProcessId) } } }",
    "$ordered = @($targets.ToArray())",
    ...(command === "Suspend" ? ["[array]::Reverse($ordered)"] : []),
    "$applied = [System.Collections.Generic.List[int]]::new()",
    "try { foreach ($pidValue in $ordered) { $target = Get-Process -Id $pidValue -ErrorAction SilentlyContinue; if ($target) { $status = [OsAiNativeProcessControl]::" +
      nativeMethod +
      '($target.Handle); if ($status -ne 0) { throw "process control failed: $pidValue" }; $applied.Add($pidValue) } } }',
    ...(command === "Suspend"
      ? [
          "catch { foreach ($pidValue in $applied) { $target = Get-Process -Id $pidValue -ErrorAction SilentlyContinue; if ($target) { [void][OsAiNativeProcessControl]::NtResumeProcess($target.Handle) } }; exit 1 }",
        ]
      : ["catch { exit 1 }"]),
  ].join("\n");
  const control = spawn(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", script],
    { windowsHide: true, stdio: ["ignore", "ignore", "pipe"] },
  );
  let detail = "";
  control.stderr?.on("data", (chunk: Buffer) => {
    detail = (detail + chunk.toString("utf8")).slice(-500);
  });
  const code = await new Promise<number | null>((resolve, reject) => {
    control.once("error", reject);
    control.once("close", resolve);
  });
  if (code !== 0)
    throw new Error(
      `Windows could not ${command.toLowerCase()} training${detail.trim() ? `: ${detail.trim()}` : ""}`,
    );
}

async function setChildPaused(shouldPause: boolean) {
  if (!child?.pid) throw new Error("The training process is not running");
  if (process.platform === "win32") {
    await runWindowsProcessControl(shouldPause ? "Suspend" : "Resume");
    return;
  }
  const signal = shouldPause ? "SIGSTOP" : "SIGCONT";
  try {
    process.kill(-child.pid, signal);
  } catch {
    child.kill(signal);
  }
}

async function requestPause() {
  if (paused || pauseGeneration || stopping || !child?.pid) return;
  if (!job.checkpointRequestPath)
    throw new Error("This session cannot save a checkpoint before pausing");
  if (autoStopGeneration) {
    pauseGeneration = autoStopGeneration;
    autoStopGeneration = "";
    autoStopRequestedAt = 0;
    state.status = "pausing";
    state.autoStopMessage = "Manual pause requested during Auto stop";
    state.message = "Saving the pending checkpoint before pausing";
    await scheduleStateWrite(true);
    return;
  }
  pauseGeneration = randomUUID();
  const pending = `${job.checkpointRequestPath}.${process.pid}.pending`;
  try {
    await fs.writeFile(pending, `${pauseGeneration}\n`, { mode: 0o600 });
    await fs.rename(pending, job.checkpointRequestPath);
  } catch (error) {
    pauseGeneration = "";
    await fs.rm(pending, { force: true }).catch(() => undefined);
    throw error;
  }
  state.status = "pausing";
  state.checkpointStatus = "requested";
  state.message = "Saving a checkpoint at the next safe training step";
  await scheduleStateWrite(true);
}

async function requestAutomaticStop(reason: string) {
  if (stopping || autoStopGeneration || pauseGeneration || !child?.pid) return;
  if (!job.checkpointRequestPath) return;
  autoStopGeneration = randomUUID();
  autoStopRequestedAt = Date.now();
  const pending = `${job.checkpointRequestPath}.${process.pid}.pending`;
  try {
    await fs.writeFile(pending, `${autoStopGeneration}\n`, { mode: 0o600 });
    await fs.rename(pending, job.checkpointRequestPath);
    state.checkpointStatus = "requested";
    state.autoStopMessage = `Loss rise detected: ${reason}. Saving the latest checkpoint`;
    state.message = "Auto stop is saving a checkpoint before stopping";
    await scheduleStateWrite(true);
  } catch (error) {
    autoStopGeneration = "";
    autoStopRequestedAt = 0;
    disableAutoStop(
      "Auto stop could not request a checkpoint; training continues",
    );
    state.status = "running";
    state.error = error instanceof Error ? error.message : String(error);
    await fs.rm(pending, { force: true }).catch(() => undefined);
    await scheduleStateWrite(true);
  }
}

async function completeAutomaticStop(savedPath: string) {
  if (!autoStopGeneration || stopping || !child?.pid) return;
  autoStopGeneration = "";
  autoStopRequestedAt = 0;
  try {
    const snapshot = await fs.stat(savedPath);
    if (!snapshot.isFile() || snapshot.size === 0)
      throw new Error("The Auto stop checkpoint adapter is empty");
  } catch (error) {
    disableAutoStop(
      "Auto stop could not verify its checkpoint; training continues",
    );
    state.status = "running";
    state.checkpointStatus = "failed";
    state.error = error instanceof Error ? error.message : String(error);
    await scheduleStateWrite(true);
    return;
  }
  autoStopped = true;
  state.autoStopMessage =
    "Stopped after sustained rising loss; checkpoint saved";
  await terminateTree().catch(async (error) => {
    state.error = error instanceof Error ? error.message : String(error);
    child?.kill("SIGKILL");
    await scheduleStateWrite(true);
  });
}

async function completeStopCheckpoint() {
  if (!stopGeneration || !stopping || !child?.pid) return;
  stopGeneration = "";
  legacyStop = true;
  if (stopTimer) clearTimeout(stopTimer);
  state.message = "Adapter and merged model saved; stopping the trainer";
  await scheduleStateWrite(true);
  await killTree();
}

async function recoverStalledAutomaticStop() {
  if (
    !autoStopGeneration ||
    !autoStopRequestedAt ||
    stopping ||
    Date.now() - autoStopRequestedAt < autoStopCheckpointTimeoutMs
  )
    return;
  autoStopGeneration = "";
  autoStopRequestedAt = 0;
  const savedPath = state.checkpointPath;
  if (savedPath) {
    const resolved = path.resolve(savedPath);
    const relative = path.relative(job.sessionDirectory, resolved);
    const snapshot = await fs.stat(resolved).catch(() => null);
    if (
      !relative.startsWith("..") &&
      !path.isAbsolute(relative) &&
      /[\\/](?:outputs|\.internal)[\\/]checkpoint[\\/]/i.test(resolved) &&
      snapshot?.isFile() &&
      snapshot.size > 0
    ) {
      autoStopped = true;
      autoStoppedAtPriorCheckpoint = true;
      state.checkpointStatus = "saved";
      state.autoStopMessage =
        "The latest checkpoint stalled; stopped with the previous saved checkpoint";
      await terminateTree().catch(async (error) => {
        state.error = error instanceof Error ? error.message : String(error);
        child?.kill("SIGKILL");
        await scheduleStateWrite(true);
      });
      return;
    }
  }
  state.checkpointStatus = "failed";
  state.message = "Auto stop could not verify a checkpoint; training continues";
  disableAutoStop(
    "The checkpoint save stalled and no previous checkpoint was found",
  );
  await scheduleStateWrite(true);
}

async function completePause(savedPath: string) {
  if (!pauseGeneration || paused || stopping || !child?.pid) return;
  try {
    const snapshot = await fs.stat(savedPath);
    if (!snapshot.isFile() || snapshot.size === 0)
      throw new Error("The checkpoint adapter is empty");
    if (child.exitCode !== null) return;
    pausedIndeterminate = state.indeterminate;
    await setChildPaused(true);
    paused = true;
    pauseGeneration = "";
    state.status = "paused";
    state.indeterminate = false;
    state.message = "Paused after saving the checkpoint";
    await scheduleStateWrite(true);
  } catch (error) {
    pauseGeneration = "";
    state.status = "running";
    state.checkpointStatus = "failed";
    state.message = "Pause could not verify the checkpoint; training continues";
    state.error = error instanceof Error ? error.message : String(error);
    await scheduleStateWrite(true);
  }
}

async function resumeTraining() {
  if (!paused || stopping || !child?.pid) return;
  await setChildPaused(false);
  paused = false;
  state.status = "running";
  state.indeterminate = pausedIndeterminate;
  state.message = "Training resumed";
  await scheduleStateWrite(true);
}

let checkingControls = false;

async function exists(file: string) {
  return fs
    .access(file)
    .then(() => true)
    .catch(() => false);
}

async function checkControlRequests() {
  if (checkingControls || finalised) return;
  checkingControls = true;
  const pausePath =
    job.pausePath || path.join(job.sessionDirectory, "pause.request");
  const resumePath =
    job.resumePath || path.join(job.sessionDirectory, "resume.request");
  try {
    if (await exists(job.stopPath)) {
      await terminateTree();
      return;
    }
    if (job.autoStopPath && (await exists(job.autoStopPath))) {
      const setting = (await fs.readFile(job.autoStopPath, "utf8")).trim();
      if (setting === "on" || setting === "off") {
        const enabled = setting === "on";
        if (state.autoStopEnabled !== enabled) {
          state.autoStopEnabled = enabled;
          if (state.request) state.request.autoStop = enabled;
          const canceledPendingStop = !enabled && Boolean(autoStopGeneration);
          if (canceledPendingStop) {
            autoStopGeneration = "";
            autoStopRequestedAt = 0;
            state.checkpointStatus = state.checkpointPath ? "saved" : "failed";
            state.message = "Auto stop canceled; training continues";
          }
          state.autoStopMessage = enabled
            ? "Gathering a loss baseline"
            : canceledPendingStop
              ? "Auto stop canceled; the requested checkpoint may still finish"
              : "Auto stop is off";
          lossRise.setEnabled(enabled);
          await scheduleStateWrite(true);
        }
      }
    }
    if (await exists(pausePath)) {
      await fs.rm(pausePath, { force: true });
      await requestPause();
    }
    if (await exists(resumePath)) {
      await fs.rm(resumePath, { force: true });
      await resumeTraining();
    }
    if (
      job.checkpointRequestPath &&
      (await exists(job.checkpointRequestPath))
    ) {
      const generation = (
        await fs.readFile(job.checkpointRequestPath, "utf8")
      ).trim();
      if (generation && generation !== lastCheckpointRequest) {
        lastCheckpointRequest = generation;
        state.checkpointStatus = "requested";
        await scheduleStateWrite(true);
      }
    }
    await recoverStalledAutomaticStop();
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    state.message = "The requested training control could not be applied";
    await scheduleStateWrite(true);
  } finally {
    checkingControls = false;
  }
}

async function terminateTree() {
  if (stopping) return;
  stopping = true;
  pauseGeneration = "";
  state.status = "stopping";
  state.message = autoStopped
    ? autoStoppedAtPriorCheckpoint
      ? "Auto stop is stopping training with the previous saved checkpoint"
      : "Auto stop saved a checkpoint and is stopping training"
    : supportsExactStop()
      ? "Saving an exact resume checkpoint at the next completed record"
      : "Saving an adapter and merged model before stopping";
  await scheduleStateWrite(true);
  if (!child?.pid) return;
  const stoppedWhilePaused =
    paused &&
    state.checkpointStatus === "saved" &&
    Boolean(state.checkpointModelPath);
  if (paused) {
    await setChildPaused(false).catch(() => undefined);
    paused = false;
  }
  await fs.writeFile(job.stopPath, "stop\n", { mode: 0o600 });
  if (!supportsExactStop()) {
    if (job.checkpointRequestPath && !autoStopped && !stoppedWhilePaused) {
      stopGeneration = randomUUID();
      state.checkpointStatus = "requested";
      const pending = `${job.checkpointRequestPath}.${process.pid}.pending`;
      await fs.writeFile(pending, `${stopGeneration}\n`, { mode: 0o600 });
      await fs.rename(pending, job.checkpointRequestPath);
      stopTimer = setTimeout(
        () => void forceStopAfterTimeout(),
        gracefulStopTimeoutMs,
      );
      stopTimer.unref();
      await scheduleStateWrite(true);
      return;
    }
    legacyStop = true;
    await killTree();
    return;
  }
  stopTimer = setTimeout(
    () => void forceStopAfterTimeout(),
    gracefulStopTimeoutMs,
  );
  stopTimer.unref();
}

async function forceStopAfterTimeout() {
  if (!child?.pid || child.exitCode !== null || finalised) return;
  forcedStop = true;
  state.error = "The trainer did not confirm a safe stop within ten minutes";
  state.message = "The trainer did not finish its requested checkpoint";
  await scheduleStateWrite(true);
  await killTree();
}

async function killTree() {
  if (!child?.pid || child.exitCode !== null || finalised) return;
  if (process.platform === "win32") {
    const killer = spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore",
    });
    const code = await new Promise<number | null>((resolve) => {
      let settled = false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const finish = (result: number | null) => {
        if (settled) return;
        settled = true;
        if (timer) clearTimeout(timer);
        resolve(result);
      };
      killer.once("error", () => finish(null));
      killer.once("close", finish);
      timer = setTimeout(() => {
        killer.kill();
        finish(null);
      }, 2_000);
      timer.unref();
    });
    if (code !== 0 || child.exitCode === null) {
      child.kill("SIGKILL");
    }
  } else {
    try {
      process.kill(-child.pid, "SIGTERM");
    } catch {
      child.kill("SIGTERM");
    }
    const timeout = setTimeout(() => {
      if (!child?.pid) return;
      try {
        process.kill(-child.pid, "SIGKILL");
      } catch {
        child.kill("SIGKILL");
      }
    }, 8_000);
    timeout.unref();
  }
}

async function finish(
  status: "completed" | "failed" | "stopped",
  code: number | null,
  error?: string,
) {
  if (finalised) return;
  finalised = true;
  if (stopTimer) clearTimeout(stopTimer);
  if (writeTimer) clearTimeout(writeTimer);
  state.status = status;
  state.phase = status === "completed" ? "complete" : state.phase;
  state.progress = status === "completed" ? 100 : state.progress;
  state.indeterminate = false;
  state.exitCode = code;
  state.endedAt = new Date().toISOString();
  if (status === "stopped") {
    const snapshot = path.join(
      job.sessionDirectory,
      ".internal",
      "checkpoint",
      "adapter",
      "last.gguf.resume",
    );
    const legacySnapshot = path.join(
      job.sessionDirectory,
      "outputs",
      "checkpoint",
      "adapter",
      "last.gguf.resume",
    );
    if ((await fs.stat(snapshot).catch(() => null))?.isFile()) {
      state.resumeCheckpointPath = snapshot;
    } else if ((await fs.stat(legacySnapshot).catch(() => null))?.isFile()) {
      state.resumeCheckpointPath = legacySnapshot;
    }
  }
  state.message =
    status === "completed"
      ? "Training completed successfully"
      : status === "stopped"
        ? autoStopped
          ? autoStoppedAtPriorCheckpoint
            ? "Auto stop used the previous saved checkpoint after the latest save stalled"
            : "Auto stop saved a checkpoint after sustained rising loss"
          : state.resumeCheckpointPath
            ? "Training stopped; exact resume checkpoint is ready"
            : state.checkpointStatus === "saved" && state.checkpointModelPath
              ? "Training stopped; adapter and merged model are ready"
              : "Training stopped by the user"
        : "Training failed; open the log for details";
  if (error) state.error = error.slice(0, 1000);
  await metricWrites.catch(() => undefined);
  await atomicStateWrite();
}

async function main() {
  job = JSON.parse(await fs.readFile(jobPath, "utf8")) as WorkerJob;
  state = JSON.parse(await fs.readFile(job.statePath, "utf8")) as SessionState;
  if (job.schemaVersion !== 1 || state.id !== job.id)
    throw new Error("Invalid training job");
  fineTuneProgress = new FineTuneProgressParser(job.iterations);
  lossRise.setEnabled(Boolean(state.autoStopEnabled));
  state.status = "running";
  state.startedAt = new Date().toISOString();
  state.workerPid = process.pid;
  state.phase = "preparing";
  state.progress = 1;
  state.indeterminate = true;
  state.message = "Checking the model and local training backend";
  await atomicStateWrite();
  if (!job.args.includes("--resume-session")) {
    await fs.writeFile(
      path.join(job.sessionDirectory, "metrics.csv"),
      metricHeader,
      { encoding: "utf8", mode: 0o600 },
    );
  }
  const log = createWriteStream(job.logPath, { flags: "a", mode: 0o600 });
  log.write(`[osAi App] ${state.startedAt}\n[osAi App] ${state.command}\n\n`);
  const backendEnvironment = await backendRuntimeEnvironment(job.executable);
  child = spawn(job.executable, job.args, {
    cwd: job.sessionDirectory,
    shell: false,
    detached: process.platform !== "win32",
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...backendEnvironment,
      PYTHONUNBUFFERED: "1",
      PYTHONIOENCODING: "utf-8",
      OSAI_APP_SESSION: job.id,
      OSAI_STOP_REQUEST: job.stopPath,
      ...(job.checkpointRequestPath
        ? { OSAI_CHECKPOINT_REQUEST: job.checkpointRequestPath }
        : {}),
    },
  });
  child.stdout?.pipe(log, { end: false });
  child.stderr?.pipe(log, { end: false });
  child.stdout?.on("data", lineReader(consumeLine));
  child.stderr?.on("data", lineReader(consumeStderrLine));
  child.once("spawn", () => {
    state.processPid = child?.pid;
    void scheduleStateWrite(true);
  });
  let completing = false;
  const closeLog = () =>
    new Promise<void>((resolve) => {
      let settled = false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const done = () => {
        if (settled) return;
        settled = true;
        if (timer) clearTimeout(timer);
        resolve();
      };
      child?.stdout?.unpipe(log);
      child?.stderr?.unpipe(log);
      log.once("error", done);
      log.end(done);
      timer = setTimeout(() => {
        log.destroy();
        done();
      }, 1_000);
      timer.unref();
    });
  const completion = new Promise<void>((resolve, reject) => {
    const complete = (
      status: "completed" | "failed" | "stopped",
      code: number | null,
      error?: string,
    ) => {
      if (completing) return;
      completing = true;
      void closeLog()
        .then(() => finish(status, code, error))
        .then(resolve, reject);
    };
    child?.once("error", (error) => {
      log.write(`\n[osAi App] ${error.message}\n`);
      complete("failed", null, error.message);
    });
    child?.once("close", async (code, signal) => {
      const stopRequested = stopping || (await exists(job.stopPath));
      const status = forcedStop
        ? "failed"
        : stopRequested && (code === 0 || legacyStop)
          ? "stopped"
          : code === 0
            ? "completed"
            : "failed";
      const error =
        status === "failed"
          ? actionableStderrLine ||
            lastStderrLine ||
            `osAi exited with ${code ?? signal ?? "an error"}`
          : undefined;
      complete(status, code, error);
    });
  });
  const stopPoll = setInterval(() => {
    void checkControlRequests().catch((error) => {
      console.error("osAi worker control check failed:", error);
    });
  }, 300);
  stopPoll.unref();
  process.on("SIGTERM", () => void terminateTree());
  process.on("SIGINT", () => void terminateTree());
  await completion;
  clearInterval(stopPoll);
}

void main().catch(async (error) => {
  if (state && job)
    await finish(
      "failed",
      null,
      error instanceof Error ? error.message : String(error),
    );
  process.exitCode = 1;
});
