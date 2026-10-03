import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { AutoCalibrationResult, TrainingRequest } from "../types.js";
import { autoBenchmarkArgs } from "./auto-benchmark.js";
import { backendRuntimeEnvironment } from "./backend-source.js";

const outputLimit = 64 * 1024;
let active: ChildProcess | null = null;
let starting = false;
let abortRequested = false;

export function autoCalibrationArgs(input: TrainingRequest) {
  if (!path.isAbsolute(input.fineTuneData))
    throw new Error("Choose a fine-tuning dataset before calibration");
  if (!["auto", "sgd", "adamw"].includes(input.optimizer))
    throw new Error("Invalid calibration optimizer");
  const args = [
    "calibrate",
    ...autoBenchmarkArgs(input).slice(2),
    "--data",
    path.resolve(input.fineTuneData),
    "--optimizer",
    input.optimizer,
  ];
  args.push(
    input.fullContentContext
      ? "--full-content-context"
      : "--no-full-content-context",
  );
  if (input.scale !== null) args.push("--scale", String(input.scale));
  if (input.dropout !== null) args.push("--dropout", String(input.dropout));
  if (input.seed !== null) args.push("--seed", String(input.seed));
  if (input.gradientAccumulationSteps !== null)
    args.push(
      "--grad-accumulation-steps",
      String(input.gradientAccumulationSteps),
    );
  if (!input.gradientCheckpoint) args.push("--no-gradient-checkpointing");
  if (!input.maskPrompt) args.push("--no-mask-prompt");
  if (input.splitMode !== "auto") args.push("--split-mode", input.splitMode);
  if (input.tensorSplit.trim())
    args.push("--tensor-split", input.tensorSplit.trim());
  if (input.mainGpu !== null) args.push("--main-gpu", String(input.mainGpu));
  if (input.distributedWorkers !== null)
    args.push("--distributed-workers", String(input.distributedWorkers));
  return args;
}

export function cancelAutoCalibration() {
  abortRequested = true;
  const child = active;
  if (!child?.pid) return;
  if (process.platform === "win32") {
    const stopper = spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore",
    });
    stopper.on("error", () => child.kill());
  } else {
    try {
      process.kill(-child.pid, "SIGTERM");
    } catch {
      child.kill();
    }
  }
}

export function validateAutoCalibrationResult(
  value: unknown,
): AutoCalibrationResult {
  const result = value as AutoCalibrationResult | null;
  if (
    !result ||
    !result.settings ||
    !Number.isInteger(result.settings.max_seq_length) ||
    result.settings.max_seq_length <= 0 ||
    !Number.isFinite(result.learning_rate) ||
    result.learning_rate <= 0 ||
    !Number.isFinite(result.first_loss) ||
    !Number.isFinite(result.last_loss) ||
    result.last_loss >= result.first_loss ||
    !Number.isFinite(result.improvement_percent) ||
    result.improvement_percent <= 0
  )
    throw new Error("osAi CLI returned invalid calibration measurements");
  return result;
}

export async function runAutoCalibration(
  executable: string,
  input: TrainingRequest,
  onProgress: (message: string) => void,
): Promise<AutoCalibrationResult> {
  if (active || starting) throw new Error("Calibration is already running");
  starting = true;
  abortRequested = false;
  let temporaryDirectory = "";
  try {
    const args = autoCalibrationArgs(input);
    const environment = await backendRuntimeEnvironment(executable);
    temporaryDirectory = await fs.mkdtemp(
      path.join(os.tmpdir(), "osai-calibration-"),
    );
    environment.OSAI_CALIBRATION_PARENT = temporaryDirectory;
    environment.OSAI_CALIBRATION_QUICK = "1";
    if (abortRequested) throw new Error("Calibration cancelled");
    return await new Promise<AutoCalibrationResult>((resolve, reject) => {
      let stdout = "";
      let stderr = "";
      let pendingLine = "";
      let settled = false;
      let lastNativeProgressAt = 0;
      const child = spawn(executable, args, {
        shell: false,
        windowsHide: true,
        detached: process.platform !== "win32",
        stdio: ["ignore", "pipe", "pipe"],
        env: environment,
      });
      active = child;
      const finish = (error?: Error, result?: AutoCalibrationResult) => {
        if (settled) return;
        settled = true;
        if (active === child) active = null;
        if (error) reject(error);
        else resolve(result!);
      };
      child.stdout?.on("data", (chunk: Buffer) => {
        stdout = (stdout + chunk.toString("utf8")).slice(-outputLimit);
      });
      child.stderr?.on("data", (chunk: Buffer) => {
        const text = chunk.toString("utf8");
        stderr = (stderr + text).slice(-outputLimit);
        pendingLine = (pendingLine + text).slice(-outputLimit);
        const lines = pendingLine.split(/\r?\n/);
        pendingLine = lines.pop() || "";
        for (const line of lines) {
          const status = /^osai: calibration phase=[\w-]+ detail=(.+)$/.exec(
            line,
          );
          if (status) onProgress(status[1].slice(0, 180));
          const scan = /^osai: full content scan records=(\d+)$/.exec(line);
          if (scan) onProgress(`Scanning training data: ${scan[1]} records`);
          if (!status && !scan && Date.now() - lastNativeProgressAt >= 1_000) {
            const epoch = /\bepoch\s+(\d+)\b/i.exec(line);
            const batch = /\bdata=(\d+)\/(\d+)\b/i.exec(line);
            if (epoch || batch) {
              const detail = epoch
                ? `Pilot training · epoch ${Number(epoch[1]) + 1}`
                : `Pilot training · batch ${Number(batch![1])} of ${Number(batch![2])}`;
              onProgress(detail);
              lastNativeProgressAt = Date.now();
            }
          }
        }
      });
      child.once("error", (error) => finish(error));
      child.once("close", (code) => {
        if (abortRequested) {
          finish(new Error("Calibration cancelled"));
          return;
        }
        if (code !== 0) {
          finish(
            new Error(
              stderr.trim().slice(-1000) || `Calibration exited with ${code}`,
            ),
          );
          return;
        }
        try {
          const result = validateAutoCalibrationResult(JSON.parse(stdout));
          finish(undefined, result);
        } catch (error) {
          finish(error instanceof Error ? error : new Error(String(error)));
        }
      });
    });
  } finally {
    starting = false;
    if (temporaryDirectory)
      await fs.rm(temporaryDirectory, {
        recursive: true,
        force: true,
        maxRetries: 20,
        retryDelay: 500,
      });
  }
}
