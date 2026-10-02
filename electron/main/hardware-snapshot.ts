import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
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

function metricNumber(value: string | undefined): number | null {
  const match = /^\s*(-?\d+(?:\.\d+)?)/.exec(value || "");
  return match ? Number(match[1]) : null;
}

function memoryBytes(
  value: string | undefined,
  defaultUnit = "MB",
): number | null {
  const amount = metricNumber(value);
  if (amount === null) return null;
  const unit =
    /\b(KiB|MiB|GiB|KB|MB|GB|B)\b/i.exec(value || "")?.[1] || defaultUnit;
  const multiplier: Record<string, number> = {
    B: 1,
    KB: 1_000,
    KIB: 1024,
    MB: 1_000_000,
    MIB: 1024 ** 2,
    GB: 1_000_000_000,
    GIB: 1024 ** 3,
  };
  return amount * multiplier[unit.toUpperCase()];
}

export function parseAmdMonitorCsv(report: string): GpuSnapshot[] {
  const rows = report
    .trim()
    .split(/\r?\n/)
    .map((line) => line.split(",").map((value) => value.trim()));
  const headerIndex = rows.findIndex((row) =>
    row.some((column) => column.toUpperCase() === "GPU"),
  );
  if (headerIndex < 0) return [];
  const headers = rows[headerIndex].map((value) =>
    value.toUpperCase().replace(/[^A-Z0-9]/g, "_"),
  );
  const field = (row: string[], names: string[]) => {
    const index = headers.findIndex((header) => names.includes(header));
    return index < 0 ? undefined : row[index];
  };
  return rows.slice(headerIndex + 1).flatMap((row) => {
    const index = metricNumber(field(row, ["GPU"]));
    if (index === null || !Number.isInteger(index)) return [];
    const usage = field(row, ["VRAM_USAGE", "GTT_USAGE"]);
    const parts = usage?.split("/");
    const sharedUnit =
      /\b(KiB|MiB|GiB|KB|MB|GB|B)\b/i.exec(parts?.[1] || "")?.[1] || "MB";
    return [
      {
        id: `amd-${index}`,
        name: `AMD GPU ${index}`,
        backend: "AMD SMI",
        memoryTotalBytes: memoryBytes(
          field(row, ["VRAM_TOTAL"]) || parts?.[1],
          sharedUnit,
        ),
        memoryUsedBytes: memoryBytes(
          field(row, ["VRAM_USED"]) || parts?.[0],
          sharedUnit,
        ),
        utilizationPercent:
          metricNumber(field(row, ["GFX_UTIL", "GFX_"])) ?? null,
        temperatureC: metricNumber(field(row, ["GPU_TEMP", "GPU_T"])) ?? null,
        note: `AMD SMI GPU ${index}`,
      },
    ];
  });
}

async function amdGpus(): Promise<GpuSnapshot[]> {
  return parseAmdMonitorCsv(await output("amd-smi", ["monitor", "--csv"]));
}

export function mergeAmdTelemetry(
  gpus: GpuSnapshot[],
  telemetry: GpuSnapshot[],
) {
  const identified = gpus.filter((gpu) =>
    /\bAMD\b|Radeon|Instinct/i.test(gpu.name),
  );
  if (identified.length !== telemetry.length) {
    gpus.push(...telemetry);
    return;
  }
  for (const [index, reading] of telemetry.entries()) {
    Object.assign(identified[index], {
      memoryTotalBytes: reading.memoryTotalBytes,
      memoryUsedBytes: reading.memoryUsedBytes,
      utilizationPercent: reading.utilizationPercent,
      temperatureC: reading.temperatureC,
      note: reading.note,
    });
  }
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

export function parseMetalGpus(
  report: string,
  architecture: string,
  systemMemoryBytes: number,
): GpuSnapshot[] {
  try {
    const entries = JSON.parse(report).SPDisplaysDataType as Record<
      string,
      unknown
    >[];
    if (!Array.isArray(entries)) return [];
    return entries.map((item, index) => {
      const name = String(item.sppci_model || item._name || `GPU ${index + 1}`);
      const vram = String(
        item.spdisplays_vram || item.spdisplays_vram_shared || "",
      );
      const amount = /([\d.]+)\s*(GB|MB)/i.exec(vram);
      const multiplier =
        amount?.[2].toUpperCase() === "GB" ? 1024 ** 3 : 1024 ** 2;
      const unified =
        architecture === "arm64" &&
        (Boolean(item.spdisplays_vram_shared) ||
          /shared|unified/i.test(vram) ||
          /Apple/i.test(name));
      const external = /external|eGPU/i.test(JSON.stringify(item));
      return {
        id: `metal-${index}`,
        name,
        backend: "Metal",
        memoryTotalBytes: unified
          ? systemMemoryBytes
          : amount
            ? Number(amount[1]) * multiplier
            : null,
        // Unified memory is shared with the CPU; host use is not GPU use.
        memoryUsedBytes: null,
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

async function metalGpus(): Promise<GpuSnapshot[]> {
  const report = await output(
    "system_profiler",
    ["SPDisplaysDataType", "-json"],
    8_000,
  );
  return parseMetalGpus(report, process.arch, os.totalmem());
}

async function linuxCpuTemperature(): Promise<number | null> {
  try {
    const root = "/sys/class/hwmon";
    const sensors = await fs.readdir(root);
    const readings: number[] = [];
    for (const sensor of sensors) {
      const directory = path.join(root, sensor);
      const name = (
        await fs.readFile(path.join(directory, "name"), "utf8")
      ).trim();
      if (!/^(coretemp|k10temp|zenpower|cpu_thermal|soc_thermal)$/i.test(name))
        continue;
      for (const file of await fs.readdir(directory)) {
        if (!/^temp\d+_input$/.test(file)) continue;
        const value =
          Number(
            (await fs.readFile(path.join(directory, file), "utf8")).trim(),
          ) / 1000;
        if (Number.isFinite(value) && value >= 0 && value <= 125)
          readings.push(value);
      }
    }
    return readings.length ? Math.max(...readings) : null;
  } catch {
    return null;
  }
}

export async function hardwareSnapshot(): Promise<HardwareSnapshot> {
  const gpus: GpuSnapshot[] = [];
  if (process.platform === "darwin") {
    mergeGpuInventory(gpus, await metalGpus());
  } else {
    const [nvidia, vulkan, system, amd] = await Promise.all([
      nvidiaGpus(),
      vulkanGpus(),
      process.platform === "win32" ? windowsGpus() : Promise.resolve([]),
      amdGpus(),
    ]);
    mergeGpuInventory(gpus, system);
    mergeGpuInventory(gpus, vulkan);
    mergeGpuInventory(gpus, nvidia);
    mergeAmdTelemetry(gpus, amd);
  }
  return {
    sampledAt: new Date().toISOString(),
    cpu: {
      name: os.cpus()[0]?.model || "CPU",
      logicalCores: os.cpus().length,
      utilizationPercent: cpuUsage(),
      memoryTotalBytes: os.totalmem(),
      memoryUsedBytes: os.totalmem() - os.freemem(),
      temperatureC:
        process.platform === "linux" ? await linuxCpuTemperature() : null,
    },
    gpus,
  };
}
