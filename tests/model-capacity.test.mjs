import assert from "node:assert/strict";
import test from "node:test";
import { modelFitsShardedGpuMemory } from "../dist-electron/model-capacity.js";

const gib = 1024 ** 3;

test("combined small cards can fit a sharded model", () => {
  assert.equal(modelFitsShardedGpuMemory(2.97 * gib, [3 * gib, 3 * gib]), true);
  assert.equal(modelFitsShardedGpuMemory(4.85 * gib, [4 * gib, 4 * gib]), true);
  assert.equal(
    modelFitsShardedGpuMemory(4.85 * gib, [4 * gib, 4 * gib, 4 * gib]),
    true,
  );
});

test("capacity is unknown instead of pessimistically disabling models", () => {
  assert.equal(modelFitsShardedGpuMemory(3 * gib, []), null);
  assert.equal(modelFitsShardedGpuMemory(3 * gib, [0, Number.NaN]), null);
});

test("a model is disabled only when combined safe capacity is insufficient", () => {
  assert.equal(
    modelFitsShardedGpuMemory(4.85 * gib, [3 * gib, 3 * gib]),
    false,
  );
});
