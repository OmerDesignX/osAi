import assert from "node:assert/strict";
import test from "node:test";
import {
  mergeAmdTelemetry,
  mergeGpuInventory,
  parseAmdMonitorCsv,
  parseMetalGpus,
  parseNvidiaGpus,
  parseVulkanGpus,
} from "../dist-electron/main/hardware-snapshot.js";

test("hardware sheet keeps distinct GPUs with identical model names", () => {
  const vulkan = parseVulkanGpus(`Devices:
GPU0:
  deviceName = AMD Radeon(TM) Graphics
GPU1:
  deviceName = NVIDIA GeForce RTX 3060
GPU2:
  deviceName = NVIDIA GeForce RTX 3060
`);
  const nvidia = parseNvidiaGpus(
    "0, NVIDIA GeForce RTX 3060, 12288, 9758, 1, 50\n" +
      "1, NVIDIA GeForce RTX 3060, 12288, 6875, 100, 51\n",
  );
  const devices = [];
  mergeGpuInventory(devices, vulkan);
  mergeGpuInventory(devices, nvidia);
  assert.equal(devices.length, 3);
  assert.equal(devices.filter((gpu) => gpu.backend === "CUDA").length, 2);
  assert.equal(devices.filter((gpu) => gpu.backend === "Vulkan").length, 1);
  assert.deepEqual(
    devices
      .filter((gpu) => gpu.backend === "CUDA")
      .map((gpu) => gpu.temperatureC),
    [50, 51],
  );
});

test("Metal inventory lists Apple silicon and Intel Mac external GPUs without inventing usage", () => {
  const apple = parseMetalGpus(
    JSON.stringify({
      SPDisplaysDataType: [
        { sppci_model: "Apple M3 Max", spdisplays_vram_shared: "36 GB" },
      ],
    }),
    "arm64",
    36 * 1024 ** 3,
  );
  assert.equal(apple[0].backend, "Metal");
  assert.equal(apple[0].memoryTotalBytes, 36 * 1024 ** 3);
  assert.equal(apple[0].memoryUsedBytes, null);
  assert.equal(apple[0].note, "Shared system memory");
  const intel = parseMetalGpus(
    JSON.stringify({
      SPDisplaysDataType: [
        {
          sppci_model: "AMD Radeon Pro 580",
          spdisplays_vram: "8 GB",
          spdisplays_bus: "eGPU",
        },
      ],
    }),
    "x64",
    32 * 1024 ** 3,
  );
  assert.equal(intel[0].memoryTotalBytes, 8 * 1024 ** 3);
  assert.equal(intel[0].note, "External GPU");
});

test("AMD SMI fills Vulkan memory, load and temperature without duplicating the GPU", () => {
  const devices = parseVulkanGpus("GPU0:\n  deviceName = AMD Radeon RX 6800\n");
  const metrics = parseAmdMonitorCsv(
    "GPU,GPU_TEMP,GFX_UTIL,VRAM_USED,VRAM_TOTAL\n0,54 C,68 %,4096 MB,16384 MB\n",
  );
  mergeAmdTelemetry(devices, metrics);
  assert.equal(devices.length, 1);
  assert.equal(devices[0].backend, "Vulkan");
  assert.equal(devices[0].temperatureC, 54);
  assert.equal(devices[0].utilizationPercent, 68);
  assert.equal(devices[0].memoryUsedBytes, 4_096_000_000);
  assert.equal(devices[0].memoryTotalBytes, 16_384_000_000);
});

test("AMD SMI accepts combined memory readings and leaves missing sensors unknown", () => {
  const [gpu] = parseAmdMonitorCsv(
    "GPU,GPU_T,GFX%,VRAM_USAGE\n0,N/A,2 %,0.3/192.0 GB\n",
  );
  assert.equal(gpu.temperatureC, null);
  assert.equal(gpu.utilizationPercent, 2);
  assert.equal(gpu.memoryUsedBytes, 300_000_000);
  assert.equal(gpu.memoryTotalBytes, 192_000_000_000);
});

test("missing sensor values remain unavailable", () => {
  const [gpu] = parseNvidiaGpus(
    "0, NVIDIA GeForce RTX 3060, 12288, N/A, N/A, N/A\n",
  );
  assert.equal(gpu.memoryUsedBytes, null);
  assert.equal(gpu.utilizationPercent, null);
  assert.equal(gpu.temperatureC, null);
});
