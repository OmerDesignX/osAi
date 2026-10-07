import assert from "node:assert/strict";
import test from "node:test";
import {
  mergeAmdTelemetry,
  mergeGpuInventory,
  parseAmdMonitorCsv,
  parseMetalGpus,
  parseNvidiaGpus,
  parseVulkanGpus,
  parseWindowsGpus,
} from "../dist-electron/main/hardware-snapshot.js";

for (const count of [3, 4, 8]) {
  test(`hardware sheet retains all ${count} identical-model GPUs`, () => {
    const devices = parseWindowsGpus(
      JSON.stringify(
        Array.from({ length: count }, (_, index) => ({
          Name: "NVIDIA GeForce GTX 980",
          PNPDeviceID: `PCI-GPU-${index}`,
          ConfigManagerErrorCode: 0,
        })),
      ),
    );
    const telemetry = parseNvidiaGpus(
      Array.from(
        { length: count },
        (_, index) =>
          `${index}, NVIDIA GeForce GTX 980, 4096, ${index * 10}, 0, ${50 + index}`,
      ).join("\n"),
    );
    mergeGpuInventory(devices, telemetry);
    assert.equal(devices.length, count);
    assert.ok(devices.every((gpu) => gpu.backend === "CUDA"));
    assert.equal(new Set(devices.map((gpu) => gpu.id)).size, count);
  });
}

test("a Windows Code 43 card stays unavailable beside identical working GPUs", () => {
  const devices = parseWindowsGpus(
    JSON.stringify([
      {
        Name: "NVIDIA GeForce GTX 980",
        PNPDeviceID: "PCI-BLOCKED",
        ConfigManagerErrorCode: 43,
      },
      {
        Name: "NVIDIA GeForce GTX 980",
        PNPDeviceID: "PCI-ONE",
        ConfigManagerErrorCode: 0,
      },
      {
        Name: "NVIDIA GeForce GTX 980",
        PNPDeviceID: "PCI-TWO",
        ConfigManagerErrorCode: 0,
      },
    ]),
  );
  mergeGpuInventory(
    devices,
    parseVulkanGpus(
      "GPU0:\n  deviceName = NVIDIA GeForce GTX 980\nGPU1:\n  deviceName = NVIDIA GeForce GTX 980\n",
    ),
  );
  mergeGpuInventory(
    devices,
    parseNvidiaGpus(
      "0, NVIDIA GeForce GTX 980, 4096, 900, 0, 55\n1, NVIDIA GeForce GTX 980, 4096, 0, 0, 51\n",
    ),
  );
  assert.equal(devices.length, 3);
  const blocked = devices.find((gpu) => gpu.driverProblemCode === 43);
  assert.equal(blocked.backend, "Unavailable");
  assert.match(blocked.note, /Code 43.*Device Manager/);
  assert.equal(blocked.memoryTotalBytes, null);
  assert.equal(blocked.utilizationPercent, null);
  assert.deepEqual(
    devices
      .filter((gpu) => gpu.backend === "CUDA")
      .map((gpu) => gpu.temperatureC),
    [55, 51],
  );
});

test("Windows inventory reports device errors without inventing GPU memory", () => {
  const [gpu] = parseWindowsGpus(
    JSON.stringify({
      Name: "AMD Radeon",
      ConfigManagerErrorCode: 22,
      AdapterRAM: 4294967295,
    }),
  );
  assert.equal(gpu.driverProblemCode, 22);
  assert.equal(gpu.memoryTotalBytes, null);
  assert.match(gpu.note, /Code 22/);
  assert.deepEqual(parseWindowsGpus("not JSON"), []);
});

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

test("Metal inventory lists Apple silicon and every physical Intel Mac GPU without inventing usage", () => {
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
          sppci_model: "AMD FirePro D700",
          spdisplays_vram: "6 GB",
          spdisplays_bus: "PCIe",
        },
        {
          sppci_model: "AMD FirePro D700",
          spdisplays_vram: "6 GB",
          spdisplays_bus: "PCIe",
        },
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
  assert.equal(intel.length, 3);
  assert.deepEqual(
    intel.map((gpu) => gpu.name),
    ["AMD FirePro D700", "AMD FirePro D700", "AMD Radeon Pro 580"],
  );
  assert.deepEqual(
    intel.map((gpu) => gpu.id),
    ["metal-0", "metal-1", "metal-2"],
  );
  assert.equal(intel[0].memoryTotalBytes, 6 * 1024 ** 3);
  assert.equal(intel[1].memoryTotalBytes, 6 * 1024 ** 3);
  assert.equal(intel[2].memoryTotalBytes, 8 * 1024 ** 3);
  assert.equal(intel[2].note, "External GPU");
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
