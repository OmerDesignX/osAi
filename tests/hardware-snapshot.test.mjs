import assert from "node:assert/strict";
import test from "node:test";
import {
  mergeGpuInventory,
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

test("missing sensor values remain unavailable", () => {
  const [gpu] = parseNvidiaGpus(
    "0, NVIDIA GeForce RTX 3060, 12288, N/A, N/A, N/A\n",
  );
  assert.equal(gpu.memoryUsedBytes, null);
  assert.equal(gpu.utilizationPercent, null);
  assert.equal(gpu.temperatureC, null);
});
