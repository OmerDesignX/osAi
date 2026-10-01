import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { SessionService } from "../dist-electron/main/session-service.js";

test("an older session gains compact loss history from its complete log", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-history-"));
  const directory = path.join(root, "saved-session");
  const id = "22222222-2222-4222-8222-222222222222";
  const logPath = path.join(directory, "training.log");
  await fs.mkdir(directory);
  await fs.writeFile(
    path.join(directory, "state.json"),
    JSON.stringify({
      schemaVersion: 1,
      id,
      name: "saved-session",
      status: "completed",
      phase: "complete",
      progress: 100,
      indeterminate: false,
      message: "complete",
      createdAt: new Date().toISOString(),
      sessionDirectory: directory,
      logPath,
      command: "osai train",
    }),
  );
  await fs.writeFile(
    logPath,
    "[CUDA0] train: data=1/100 loss=1.5±0.1 acc=50±2%\n" +
      "[CUDA0] train: data=20/100 loss=1.0±0.1 acc=70±2%\n" +
      "osai: checkpoint saved path=adapter.gguf generation=auto\n",
  );
  const service = new SessionService(root, "", async () => ({
    sessionsRoot: root,
    sessionRoots: [],
  }));
  try {
    const first = await service.metrics(id);
    assert.equal(first.filter((point) => point.event === "loss").length, 2);
    assert.equal(
      first.filter((point) => point.event === "checkpoint").length,
      1,
    );
    assert.equal(first[2].percent, 20);
    assert.equal(first[0].step, 1);
    assert.equal(first[1].step, 20);
    assert.equal(first[2].step, 20);
    await fs.appendFile(
      logPath,
      "[CUDA0] train: data=50/100 loss=0.7±0.1 acc=80±2%\n",
    );
    const resumed = await service.metrics(id);
    assert.equal(resumed.filter((point) => point.event === "loss").length, 3);
    assert.equal(
      (await fs.readFile(path.join(directory, "metrics.csv"), "utf8")).split(
        "\n",
      ).length < 10,
      true,
    );
  } finally {
    await fs.rm(root, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 100,
    });
  }
});
