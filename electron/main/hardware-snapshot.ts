import { execFile } from "node:child_process";
import os from "node:os";
import { promisify } from "node:util";
import type { HardwareSnapshot, GpuSnapshot } from "../types.js";

const execute = promisify(execFile);
let previousCpu: { idle: number; total: number } | null = null;

async function output(command: string, args: string[], timeout = 4_000) {
  try {
    return (
      await execute(command, args, {
        timeout,
        maxBuffer: 1024 * 1024,
        windowsHide: true,
      })
    ).stdout;
  } catch {
    return "";
  }
}

function cpuUsage() {
  const times = os.cpus().flatMap((cpu) => Object.entries(cpu.times));
  const current = {
    idle: times
      .filter(([name]) => name === "idle")
      .reduce((sum, [, value]) => sum + value, 0),
    total: times.reduce((sum, [, value]) => sum + value, 0),
  };
  const prior = previousCpu;
  previousCpu = current;
  if (!prior || current.total <= prior.total) return null;
  return Math.max(
    0,
    Math.min(
      100,
      (100 * (current.total - prior.total - (current.idle - prior.idle))) /
        (current.total - prior.total),
    ),
  );
}

function gpuKey(name: string) {
  return name
    .toLowerCase()
    .replace(/\b(nvidia|amd|radeon|graphics)\b/g, "")
    .replace(/[^a-z0-9]/g, "");
}

export function mergeGpuInventory(
  gpus: GpuSnapshot[],
  inventory: GpuSnapshot[],
) {
  const matched = new Set<GpuSnapshot>();
  for (const gpu of inventory) {
    const existing = gpus.find(
      (item) => gpuKey(item.name) === gpuKey(gpu.name) && !matched.has(item),
    );
    if (!existing) {
      gpus.push(gpu);
      matched.add(gpu);
    } else {
      matched.add(existing);
      if (gpu.backend === "CUDA") Object.assign(existing, gpu);
      else if (existing.backend === "Detected by OS")
        existing.backend = gpu.backend;
    }
  }
}

export function parseNvidiaGpus(report: string): GpuSnapshot[] {
  return report
    .split(/\r?\n/)
    .map((line) => line.split(",").map((value) => value.trim()))
    .filter((fields) => fields.length === 6 && /^\d+$/.test(fields[0]))
    .map(([index, name, total, used, utilization, temperature]) => ({
      id: `cuda-${index}`,
      name,
      backend: "CUDA",
      memoryTotalBytes: Number.isFinite(Number(total))
        ? Number(total) * 1024 ** 2
        : null,
      memoryUsedBytes: Number.isFinite(Number(used))
        ? Number(used) * 1024 ** 2
        : null,
      utilizationPercent: Number.isFinite(Number(utilization))
        ? Number(utilization)
        : null,
      temperatureC: Number.isFinite(Number(temperature))
        ? Number(temperature)
        : null,
      note: `GPU ${index}`,
    }));
}

async function nvidiaGpus(): Promise<GpuSnapshot[]> {
  const report = await output("nvidia-smi", [
    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
  ]);
  return parseNvidiaGpus(report);
}

export function parseVulkanGpus(report: string): GpuSnapshot[] {
  return [
    ...report.matchAll(
      /^\s*GPU(\d+):[\s\S]*?^\s*deviceName\s*=\s*(.+?)\s*$/gim,
    ),
  ].map((match) => ({
    id: `vulkan-${match[1]}`,
    name: match[2].trim(),
    backend: "Vulkan",
    memoryTotalBytes: null,
    memoryUsedBytes: null,
    utilizationPercent: null,
    temperatureC: null,
    note: `GPU ${match[1]}`,
  }));
}

async function vulkanGpus(): Promise<GpuSnapshot[]> {
  return parseVulkanGpus(await output("vulkaninfo", ["--summary"]));
}

async function windowsGpus(): Promise<GpuSnapshot[]> {
  const report = await output("powershell.exe", [
    "-NoProfile",
    "-Command",
    "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
  ]);
  try {
    const parsed = JSON.parse(report) as
      Record<string, unknown> | Record<string, unknown>[];
    return (Array.isArray(parsed) ? parsed : [parsed])
      .filter(
        (item) =>
          typeof item.Name === "string" && !/basic display/i.test(item.Name),
      )
      .map((item, index) => ({
        id: `windows-${index}`,
        name: String(item.Name),
        backend: "Detected by OS",
        // Win32_VideoController.AdapterRAM is limited to 32 bits on many drivers.
        memoryTotalBytes: null,
        memoryUsedBytes: null,
        utilizationPercent: null,
        temperatureC: null,
        note: "Live telemetry unavailable from this driver",
      }));
  } catch {
    return [];
  }
}

async function metalGpus(): Promise<GpuSnapshot[]> {
  const report = await output(
    "system_profiler",
    ["SPDisplaysDataType", "-json"],
    8_000,
  );
  try {
    const entries = JSON.parse(report).SPDisplaysDataType as Record<
      string,
      unknown
    >[];
    return entries.map((item, index) => {
      const name = String(item.sppci_model || item._name || `GPU ${index + 1}`);
      const vram = String(
        item.spdisplays_vram || item.spdisplays_vram_shared || "",
      );
      const amount = /([\d.]+)\s*(GB|MB)/i.exec(vram);
      const multiplier =
        amount?.[2].toUpperCase() === "GB" ? 1024 ** 3 : 1024 ** 2;
      const unified = process.arch === "arm64" && /shared|unified/i.test(vram);
      const external = /external|eGPU/i.test(JSON.stringify(item));
      return {
        id: `metal-${index}`,
        name,
        backend: "Metal",
        memoryTotalBytes: unified
          ? os.totalmem()
          : amount
            ? Number(amount[1]) * multiplier
            : null,
        memoryUsedBytes: unified ? os.totalmem() - os.freemem() : null,
        utilizationPercent: null,
        temperatureC: null,
        note: external
          ? "External GPU"
          : unified
            ? "Shared system memory"
            : "GPU telemetry unavailable",
      };
    });
  } catch {
    return [];
  }
}

export async function hardwareSnapshot(): Promise<HardwareSnapshot> {
  const gpus: GpuSnapshot[] = [];
  if (process.platform === "darwin") {
    mergeGpuInventory(gpus, await metalGpus());
  } else {
    const [nvidia, vulkan, system] = await Promise.all([
      nvidiaGpus(),
      vulkanGpus(),
      process.platform === "win32" ? windowsGpus() : Promise.resolve([]),
    ]);
    mergeGpuInventory(gpus, system);
    mergeGpuInventory(gpus, vulkan);
    mergeGpuInventory(gpus, nvidia);
  }
  return {
    sampledAt: new Date().toISOString(),
    cpu: {
      name: os.cpus()[0]?.model || "CPU",
      logicalCores: os.cpus().length,
      utilizationPercent: cpuUsage(),
      memoryTotalBytes: os.totalmem(),
      memoryUsedBytes: os.totalmem() - os.freemem(),
      temperatureC: null,
    },
    gpus,
  };
}
