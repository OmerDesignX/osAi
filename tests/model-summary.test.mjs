import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { modelTrainingSummary } from "../dist-electron/main/model-summary.js";

test("custom model size uses the larger available weight format", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-model-summary-"));
  try {
    await fs.mkdir(path.join(root, "gguf"));
    await fs.mkdir(path.join(root, "mlx"));
    await fs.writeFile(path.join(root, "gguf", "model.gguf"), "12345");
    await fs.writeFile(path.join(root, "mlx", "part-1.safetensors"), "1234");
    await fs.writeFile(path.join(root, "mlx", "part-2.safetensors"), "1234");
    assert.equal(await modelTrainingSummary(root), 8);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
