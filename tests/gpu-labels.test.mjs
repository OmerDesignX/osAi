import assert from "node:assert/strict";
import test from "node:test";

import {
  gpuInventoryIdentity,
  matchNativeGpuDevices,
  nativeGpuLabel,
} from "../dist-electron/gpu-labels.js";

test("native Metal IDs are displayed with their physical GPU names", () => {
  const labels = {
    MTL0: "AMD Radeon RX Vega 64",
    MTL1: "AMD Radeon RX 580",
  };
  assert.equal(nativeGpuLabel("MTL0", labels), "MTL0 · AMD Radeon RX Vega 64");
  assert.deepEqual(
    matchNativeGpuDevices(["Radeon RX 580", "Radeon RX Vega 64"], labels),
    ["MTL1", "MTL0"],
  );
});

test("identical GPU names receive distinct native IDs", () => {
  assert.deepEqual(
    matchNativeGpuDevices(["AMD FirePro D500", "AMD FirePro D500"], {
      MTL0: "AMD FirePro D500",
      MTL1: "AMD FirePro D500",
    }),
    ["MTL0", "MTL1"],
  );
});

test("hardware identity follows physical cards without reacting to temporary free memory", () => {
  const inventory = {
    accelerator: "metal",
    devices: ["MTL0", "MTL1"],
    total_bytes: { MTL0: 8, MTL1: 4 },
    free_bytes: { MTL0: 5, MTL1: 3 },
    labels: { MTL0: "AMD Radeon RX Vega 64", MTL1: "AMD Radeon RX 580" },
  };
  assert.equal(
    gpuInventoryIdentity(inventory),
    gpuInventoryIdentity({
      ...inventory,
      free_bytes: { MTL0: 2, MTL1: 1 },
    }),
  );
  assert.notEqual(
    gpuInventoryIdentity(inventory),
    gpuInventoryIdentity({
      ...inventory,
      total_bytes: { MTL0: 4, MTL1: 8 },
      labels: {
        MTL0: "AMD Radeon RX 580",
        MTL1: "AMD Radeon RX Vega 64",
      },
    }),
  );
});
