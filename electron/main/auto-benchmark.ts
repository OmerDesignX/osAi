import { spawn, type ChildProcess } from "node:child_process";
import os from "node:os";
import path from "node:path";
import type { AutoBenchmarkResult, TrainingRequest } from "../types.js";

const outputLimit = 64 * 1024;
const benchmarkTimeoutMs = 60 * 60 * 1_000;
let active: ChildProcess | null = null;

export function autoBenchmarkArgs(input: TrainingRequest) {
  const args = ["auto-benchmark", "--refresh"];
  if (input.modelSource === "official") {
    if (!["v1", "v2"].includes(input.modelVersion))
      throw new Error("Invalid model version");
    if (!["xsmall", "small", "medium", "large"].includes(input.tier))
      throw new Error("Invalid model tier");
    args.push(
      "--tier",
      input.tier,
      "--model-version",
      input.modelVersion,
      "--bundled-root",
      path.join(os.homedir(), "osAi", "models"),
    );
  } else if (input.modelSource === "custom") {
    if (!path.isAbsolute(input.customModelFolder))
      throw new Error("Choose a custom model folder before benchmarking");
    const folder = path.resolve(input.customModelFolder);
    args.push(
      "--custom",
      path.basename(folder),
      "--custom-root",
      path.dirname(folder),
    );
  } else throw new Error("Invalid model source");
  if (!["auto", "mlx", "llama.cpp"].includes(input.engine))
    throw new Error("Invalid training engine");
  if (
    !["auto", "metal", "mps", "cuda", "vulkan", "cpu"].includes(
      input.accelerator,
    )
  )
    throw new Error("Invalid accelerator");
  if (!["auto", "on", "off"].includes(input.multiGpu))
    throw new Error("Invalid multi-GPU setting");
  args.push(
    "--engine",
    input.engine,
    "--accelerator",
    input.accelerator,
    "--multi-gpu",
    input.multiGpu,
  );
  const devices = input.devices
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  if (
    devices.length > 64 ||
    devices.some((value) => value.length > 128 || /[\r\n\0]/.test(value))
  )
    throw new Error("Invalid GPU device list");
  for (const device of devices) args.push("--device", device);
  return args;
}

export function runAutoBenchmark(
  executable: string,
  input: TrainingRequest,
): Promise<AutoBenchmarkResult> {
  const args = autoBenchmarkArgs(input);
  if (active) active.kill();
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    const child = spawn(executable, args, {
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    active = child;
    const finish = (error?: Error, result?: AutoBenchmarkResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (active === child) active = null;
      if (error) reject(error);
      else resolve(result!);
    };
    const timer = setTimeout(() => {
      child.kill();
      finish(new Error("The hardware benchmark timed out"));
    }, benchmarkTimeoutMs);
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout = (stdout + chunk.toString("utf8")).slice(-outputLimit);
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr = (stderr + chunk.toString("utf8")).slice(-outputLimit);
    });
    child.once("error", (error) => finish(error));
    child.once("close", (code) => {
      if (code !== 0) {
        finish(
          new Error(
            stderr.trim().slice(-1000) ||
              `Hardware benchmark exited with ${code}`,
          ),
        );
        return;
      }
      try {
        const result = JSON.parse(stdout) as AutoBenchmarkResult;
        if (
          !result.settings ||
          !Number.isInteger(result.settings.max_seq_length)
        )
          throw new Error("osAi CLI returned invalid benchmark settings");
        finish(undefined, result);
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      }
    });
  });
}

export function readAutoDevices(
  executable: string,
  accelerator: TrainingRequest["accelerator"],
): Promise<{ accelerator: string; devices: string[] }> {
  if (!["auto", "metal", "mps", "cuda", "vulkan", "cpu"].includes(accelerator))
    return Promise.reject(new Error("Invalid accelerator"));
  return new Promise((resolve, reject) => {
    const child = spawn(
      executable,
      ["auto-devices", "--accelerator", accelerator],
      {
        shell: false,
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    let output = "";
    let error = "";
    const timer = setTimeout(() => child.kill(), 15_000);
    child.stdout?.on("data", (chunk: Buffer) => {
      output = (output + chunk.toString("utf8")).slice(-outputLimit);
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      error = (error + chunk.toString("utf8")).slice(-outputLimit);
    });
    child.once("error", reject);
    child.once("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(error.trim().slice(-500) || "GPU discovery failed"));
        return;
      }
      try {
        const result = JSON.parse(output) as {
          accelerator: string;
          devices: string[];
        };
        if (!Array.isArray(result.devices))
          throw new Error("Invalid GPU inventory");
        resolve(result);
      } catch (failure) {
        reject(failure);
      }
    });
  });
}
