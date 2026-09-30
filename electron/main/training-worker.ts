import { spawn, type ChildProcess } from "node:child_process";
import { createWriteStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import type { SessionState, WorkerJob } from "../types.js";
import { backendRuntimeEnvironment } from "./backend-source.js";
import {
  cleanTerminalLine,
  FineTuneProgressParser,
  phaseProgress,
} from "./training-progress.js";

const jobPath = process.argv[2];
if (!jobPath || !path.isAbsolute(jobPath)) process.exit(2);

let state: SessionState;
let job: WorkerJob;
let child: ChildProcess | null = null;
let stopping = false;
let paused = false;
let pausedIndeterminate = false;
let finalised = false;
let lastStderrLine = "";
let actionableStderrLine = "";
let actionableStderrScore = 0;
let writeTimer: ReturnType<typeof setTimeout> | null = null;
let writeChain = Promise.resolve();
let fineTuneProgress: FineTuneProgressParser;
let lastCheckpointRequest = "";

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
  if (writeTimer) clearTimeout(writeTimer);
  if (immediate) return atomicStateWrite();
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
      /[\\/]outputs[\\/]checkpoint[\\/]/i.test(savedPath)
    ) {
      state.checkpointStatus = "saved";
      state.checkpointSavedAt = new Date().toISOString();
      state.checkpointPath = savedPath;
      void scheduleStateWrite(true);
    }
    return;
  }
  const checkpointModel = /osai: checkpoint model ready path=(.+)$/i.exec(line);
  if (checkpointModel) {
    const modelPath = path.resolve(checkpointModel[1]);
    const relative = path.relative(job.sessionDirectory, modelPath);
    if (!relative.startsWith("..") && !path.isAbsolute(relative)) {
      state.checkpointModelPath = modelPath;
      void scheduleStateWrite(true);
    }
    return;
  }
  if (/osai: checkpoint failed\b/i.test(line)) {
    state.checkpointStatus = "failed";
    void scheduleStateWrite(true);
    return;
  }
  if (/^osai: full content scan\b/i.test(line)) {
    const records = /\brecords=(\d+)/.exec(line)?.[1];
    state.phase = "preparing";
    state.indeterminate = true;
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
    state.message = `Preparing ${attempt[1]} training attempt ${attempt[2]} with ${attempt[3]} token windows`;
    void scheduleStateWrite();
    return;
  }
  const lower = line.toLowerCase();
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
      state.progress = clamp(
        phaseProgress(completed, total, "fine-tuning", job.stage),
      );
      const percentComplete = ((100 * completed) / Math.max(1, total)).toFixed(
        2,
      );
      state.message =
        `Fine-tuning: ${completed.toLocaleString()} of ${total.toLocaleString()} ` +
        `(${percentComplete}%)`;
    } else if (lower.includes("publish") || lower.includes("fusion")) {
      state.phase = "publishing";
      state.indeterminate = true;
      state.progress = clamp(98);
      state.message = "Publishing the adapter and final model outputs";
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
  return (chunk: Buffer) => {
    pending += chunk.toString("utf8");
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
    `$target = Get-Process -Id ${child.pid} -ErrorAction Stop`,
    `$status = [OsAiNativeProcessControl]::${nativeMethod}($target.Handle)`,
    "if ($status -ne 0) { exit 1 }",
  ].join("; ");
  const control = spawn(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", script],
    { windowsHide: true, stdio: "ignore" },
  );
  const code = await new Promise<number | null>((resolve, reject) => {
    control.once("error", reject);
    control.once("close", resolve);
  });
  if (code !== 0)
    throw new Error(`Windows could not ${command.toLowerCase()} training`);
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

async function pauseTraining() {
  if (paused || stopping || !child?.pid) return;
  pausedIndeterminate = state.indeterminate;
  await setChildPaused(true);
  paused = true;
  state.status = "paused";
  state.indeterminate = false;
  state.message = "Training paused";
  await scheduleStateWrite(true);
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
    if (await exists(pausePath)) {
      await fs.rm(pausePath, { force: true });
      await pauseTraining();
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
  state.status = "stopping";
  state.message = "Stopping the osAi process";
  await scheduleStateWrite(true);
  if (!child?.pid) return;
  if (paused) {
    await setChildPaused(false).catch(() => undefined);
    paused = false;
  }
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
  if (writeTimer) clearTimeout(writeTimer);
  state.status = status;
  state.phase = status === "completed" ? "complete" : state.phase;
  state.progress = status === "completed" ? 100 : state.progress;
  state.indeterminate = false;
  state.exitCode = code;
  state.endedAt = new Date().toISOString();
  state.message =
    status === "completed"
      ? "Training completed successfully"
      : status === "stopped"
        ? "Training stopped by the user"
        : "Training failed; open the log for details";
  if (error) state.error = error.slice(0, 1000);
  await atomicStateWrite();
}

async function main() {
  job = JSON.parse(await fs.readFile(jobPath, "utf8")) as WorkerJob;
  state = JSON.parse(await fs.readFile(job.statePath, "utf8")) as SessionState;
  if (job.schemaVersion !== 1 || state.id !== job.id)
    throw new Error("Invalid training job");
  fineTuneProgress = new FineTuneProgressParser(job.iterations);
  state.status = "running";
  state.startedAt = new Date().toISOString();
  state.workerPid = process.pid;
  state.phase = "preparing";
  state.progress = 1;
  state.indeterminate = true;
  state.message = "Checking the model and local training backend";
  await atomicStateWrite();
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
      OSAI_APP_SESSION: job.id,
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
    child?.once("close", (code, signal) => {
      const status = stopping ? "stopped" : code === 0 ? "completed" : "failed";
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
