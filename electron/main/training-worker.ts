import { spawn, type ChildProcess } from "node:child_process";
import { createWriteStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import type { SessionState, WorkerJob } from "../types.js";

const jobPath = process.argv[2];
if (!jobPath || !path.isAbsolute(jobPath)) process.exit(2);

let state: SessionState;
let job: WorkerJob;
let child: ChildProcess | null = null;
let stopping = false;
let finalised = false;
let writeTimer: ReturnType<typeof setTimeout> | null = null;
let writeChain = Promise.resolve();

async function atomicStateWrite() {
  const snapshot = JSON.stringify(state, null, 2);
  writeChain = writeChain.then(async () => {
    const temporary = `${job.statePath}.${process.pid}.tmp`;
    await fs.writeFile(temporary, `${snapshot}\n`, { mode: 0o600 });
    await fs.rename(temporary, job.statePath);
  });
  await writeChain;
}

function scheduleStateWrite(immediate = false) {
  if (writeTimer) clearTimeout(writeTimer);
  if (immediate) return atomicStateWrite();
  writeTimer = setTimeout(() => {
    writeTimer = null;
    void atomicStateWrite();
  }, 250);
  writeTimer.unref();
  return Promise.resolve();
}

function clamp(value: number) {
  return Math.max(state.progress, Math.min(99, Math.round(value)));
}

function phaseProgress(
  completed: number,
  total: number,
  phase: "fine-tuning" | "alignment",
) {
  const ratio = Math.max(0, Math.min(1, completed / Math.max(1, total)));
  if (job.stage === "fine-tune-align")
    return phase === "fine-tuning" ? 5 + ratio * 58 : 72 + ratio * 25;
  return 5 + ratio * 92;
}

function consumeLine(raw: string) {
  const line = raw.replace(/\x1b\[[0-9;]*m/g, "").trim();
  if (!line) return;
  const lower = line.toLowerCase();
  const percent = /(?:^|\s)(\d{1,3})%(?:\s|$)/.exec(line);
  if (lower.includes("download") && percent) {
    state.phase = "download";
    state.indeterminate = false;
    state.progress = clamp(2 + Math.min(100, Number(percent[1])) * 0.03);
    state.message = line.slice(0, 240);
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
    const iteration =
      /\b(?:iter|iteration|epoch)\s*[=:]?\s*(\d+)(?:\s*\/\s*(\d+))?/i.exec(
        line,
      );
    if (alignment) {
      const completed = Number(alignment[1]);
      state.phase = "alignment";
      state.indeterminate = false;
      state.progress = clamp(
        phaseProgress(completed, job.alignmentIterations, "alignment"),
      );
      state.message = `Alignment update ${completed} of ${job.alignmentIterations}`;
    } else if (iteration && state.phase !== "alignment") {
      const completed = Number(iteration[1]);
      const total = Number(iteration[2] || job.iterations);
      state.phase = "fine-tuning";
      state.indeterminate = false;
      state.progress = clamp(phaseProgress(completed, total, "fine-tuning"));
      state.message = `Fine-tuning update ${completed} of ${total}`;
    } else if (lower.includes("publish") || lower.includes("fusion")) {
      state.phase = "publishing";
      state.indeterminate = true;
      state.progress = clamp(98);
      state.message = "Publishing the adapter and final model outputs";
    }
  }
  void scheduleStateWrite();
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

async function terminateTree() {
  if (stopping) return;
  stopping = true;
  state.status = "stopping";
  state.message = "Stopping the osAi process";
  await scheduleStateWrite(true);
  if (!child?.pid) return;
  if (process.platform === "win32") {
    const killer = spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore",
    });
    await new Promise<void>((resolve) => killer.once("close", () => resolve()));
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
  child = spawn(job.executable, job.args, {
    cwd: job.sessionDirectory,
    shell: false,
    detached: process.platform !== "win32",
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1", OSAI_APP_SESSION: job.id },
  });
  child.stdout?.pipe(log, { end: false });
  child.stderr?.pipe(log, { end: false });
  child.stdout?.on("data", lineReader(consumeLine));
  child.stderr?.on("data", lineReader(consumeLine));
  child.once("spawn", () => {
    state.processPid = child?.pid;
    void scheduleStateWrite(true);
  });
  child.once("error", (error) => {
    log.write(`\n[osAi App] ${error.message}\n`);
    void finish("failed", null, error.message).finally(() => log.end());
  });
  child.once("close", (code, signal) => {
    const status = stopping ? "stopped" : code === 0 ? "completed" : "failed";
    const error =
      status === "failed"
        ? `osAi exited with ${code ?? signal ?? "an error"}`
        : undefined;
    void finish(status, code, error).finally(() => log.end());
  });
  const stopPoll = setInterval(() => {
    void fs
      .access(job.stopPath)
      .then(terminateTree)
      .catch(() => undefined);
  }, 600);
  stopPoll.unref();
  process.on("SIGTERM", () => void terminateTree());
  process.on("SIGINT", () => void terminateTree());
  await new Promise<void>((resolve) => child?.once("close", () => resolve()));
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
