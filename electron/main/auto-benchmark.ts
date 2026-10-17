import { spawn, type ChildProcess } from "node:child_process";
import os from "node:os";
import path from "node:path";
import type {
  AutoBenchmarkResult,
  AutoDeviceInventory,
  TrainingRequest,
} from "../types.js";
import { backendRuntimeEnvironment } from "./backend-source.js";

const outputLimit = 64 * 1024;
const benchmarkTimeoutMs = 60 * 60 * 1_000;
export const autoDeviceDiscoveryTimeoutMs = 120_000;
let active: ChildProcess | null = null;

export function autoBenchmarkArgs(input: TrainingRequest) {
  // The CLI cache is keyed by the physical devices, available-memory class,
  // CPU, model files and native backend. Let that fingerprint decide when a
  // fresh probe is necessary instead of remeasuring unchanged hardware at
  // every app launch.
  const args = ["auto-benchmark"];
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
    new Set(devices).size !== devices.length ||
    devices.some((value) => value.length > 128 || /[\r\n\0]/.test(value))
  )
    throw new Error("GPU device list must contain distinct valid devices");
  for (const device of devices) args.push("--device", device);
  return args;
}

export function validateAutoBenchmarkResult(
  value: unknown,
): AutoBenchmarkResult {
  const result = value as AutoBenchmarkResult | null;
  if (
    !result?.settings ||
    !Number.isInteger(result.settings.max_seq_length) ||
    !Array.isArray(result.devices) ||
    !Array.isArray(result.tensor_split) ||
    result.tensor_split.some(
      (weight) => !Number.isFinite(weight) || weight <= 0,
    ) ||
    (result.tensor_split.length > 0 &&
      result.tensor_split.length !== result.devices.length) ||
    !Number.isInteger(result.main_gpu) ||
    result.main_gpu < 0 ||
    (result.devices.length > 0 && result.main_gpu >= result.devices.length)
  )
    throw new Error("osAi CLI returned invalid benchmark settings");
  return result;
}

export function validateAutoDeviceInventory(
  value: unknown,
): AutoDeviceInventory {
  const result = value as Partial<AutoDeviceInventory> | null;
  const validMemoryMap = (memory: unknown) =>
    memory === undefined ||
    (typeof memory === "object" &&
      memory !== null &&
      Object.values(memory).every(
        (amount) =>
          typeof amount === "number" && Number.isFinite(amount) && amount > 0,
      ));
  if (
    typeof result?.accelerator !== "string" ||
    !Array.isArray(result.devices) ||
    result.devices.some((device) => typeof device !== "string" || !device) ||
    !validMemoryMap(result.free_bytes) ||
    !validMemoryMap(result.total_bytes)
  )
    throw new Error("Invalid GPU inventory");
  return {
    accelerator: result.accelerator,
    devices: result.devices,
    free_bytes: result.free_bytes ?? {},
    total_bytes: result.total_bytes ?? {},
  };
}

export async function runAutoBenchmark(
  executable: string,
  input: TrainingRequest,
): Promise<AutoBenchmarkResult> {
  const args = autoBenchmarkArgs(input);
  const environment = await backendRuntimeEnvironment(executable);
  if (active) active.kill();
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    const child = spawn(executable, args, {
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: environment,
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
        const result = validateAutoBenchmarkResult(JSON.parse(stdout));
        finish(undefined, result);
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      }
    });
  });
}

export async function readAutoDevices(
  executable: string,
  accelerator: TrainingRequest["accelerator"],
): Promise<AutoDeviceInventory> {
  if (!["auto", "metal", "mps", "cuda", "vulkan", "cpu"].includes(accelerator))
    return Promise.reject(new Error("Invalid accelerator"));
  const environment = await backendRuntimeEnvironment(executable);
  return new Promise((resolve, reject) => {
    const child = spawn(
      executable,
      ["auto-devices", "--accelerator", accelerator],
      {
        shell: false,
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
        env: environment,
      },
    );
    let output = "";
    let error = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, autoDeviceDiscoveryTimeoutMs);
    child.stdout?.on("data", (chunk: Buffer) => {
      output = (output + chunk.toString("utf8")).slice(-outputLimit);
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      error = (error + chunk.toString("utf8")).slice(-outputLimit);
    });
    child.once("error", reject);
    child.once("close", (code) => {
      clearTimeout(timer);
      if (timedOut) {
        reject(
          new Error(
            "GPU discovery timed out while Metal was starting. Restart osAi and try again.",
          ),
        );
        return;
      }
      if (code !== 0) {
        reject(new Error(error.trim().slice(-500) || "GPU discovery failed"));
        return;
      }
      try {
        resolve(validateAutoDeviceInventory(JSON.parse(output)));
      } catch (failure) {
        reject(failure);
      }
    });
  });
}
