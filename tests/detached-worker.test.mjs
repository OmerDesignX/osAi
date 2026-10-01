import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

async function waitFor(file, predicate, timeout = 20_000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    try {
      const value = JSON.parse(await fs.readFile(file, "utf8"));
      if (predicate(value)) return value;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 80));
  }
  throw new Error(`Timed out waiting for ${file}`);
}

test("detached training continues after its launcher exits", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-worker-"));
  const worker = path.resolve("dist-electron/main/training-worker.js");
  const launcher = path.resolve("tests/fixtures/launch-worker.mjs");
  const backend = path.resolve("tests/fixtures/fake-backend.mjs");
  const statePath = path.join(root, "state.json");
  const logPath = path.join(root, "training.log");
  const jobPath = path.join(root, "job.json");
  const now = new Date().toISOString();
  const state = {
    schemaVersion: 1,
    id: "11111111-1111-4111-8111-111111111111",
    name: "lifecycle-test",
    status: "queued",
    phase: "preparing",
    progress: 0,
    indeterminate: true,
    message: "queued",
    createdAt: now,
    sessionDirectory: root,
    logPath,
    command: "node fake-backend.mjs",
  };
  const job = {
    schemaVersion: 1,
    id: state.id,
    executable: process.execPath,
    args: [backend],
    sessionDirectory: root,
    statePath,
    logPath,
    stopPath: path.join(root, "stop.request"),
    stage: "fine-tuning",
    iterations: 2,
    alignmentIterations: 1,
    createdAt: now,
  };
  await fs.writeFile(statePath, JSON.stringify(state));
  await fs.writeFile(jobPath, JSON.stringify(job));
  try {
    const controller = spawn(process.execPath, [launcher, worker, jobPath], {
      stdio: "ignore",
    });
    const controllerCode = await new Promise((resolve) =>
      controller.once("close", resolve),
    );
    assert.equal(controllerCode, 0);
    const complete = await waitFor(
      statePath,
      (value) => value.status === "completed",
    );
    assert.equal(complete.progress, 100);
    assert.match(await fs.readFile(logPath, "utf8"), /Iter 2\/2/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("a completed checkpoint request remains saved after worker polling", async () => {
  const root = await fs.mkdtemp(
    path.join(os.tmpdir(), "osai-worker-checkpoint-"),
  );
  const worker = path.resolve("dist-electron/main/training-worker.js");
  const backend = path.resolve("tests/fixtures/fake-backend.mjs");
  const statePath = path.join(root, "state.json");
  const jobPath = path.join(root, "job.json");
  const checkpointRequestPath = path.join(root, "checkpoint.request");
  const id = "99999999-9999-4999-8999-999999999999";
  const now = new Date().toISOString();
  await fs.writeFile(checkpointRequestPath, "save-now\n");
  await fs.writeFile(
    statePath,
    JSON.stringify({
      schemaVersion: 1,
      id,
      name: "checkpoint-test",
      status: "queued",
      phase: "preparing",
      progress: 0,
      indeterminate: true,
      message: "queued",
      createdAt: now,
      sessionDirectory: root,
      logPath: path.join(root, "training.log"),
      command: "node fake-backend.mjs --checkpoint",
    }),
  );
  await fs.writeFile(
    jobPath,
    JSON.stringify({
      schemaVersion: 1,
      id,
      executable: process.execPath,
      args: [backend, "--checkpoint"],
      sessionDirectory: root,
      statePath,
      logPath: path.join(root, "training.log"),
      stopPath: path.join(root, "stop.request"),
      checkpointRequestPath,
      stage: "fine-tuning",
      iterations: 1,
      alignmentIterations: 1,
      createdAt: now,
    }),
  );
  const processHandle = spawn(process.execPath, [worker, jobPath], {
    stdio: "ignore",
  });
  try {
    const saved = await waitFor(
      statePath,
      (value) => value.checkpointStatus === "saved",
    );
    assert.match(
      saved.checkpointPath,
      /[\\/]outputs[\\/]checkpoint[\\/]adapter[\\/]last\.gguf$/,
    );
    await new Promise((resolve) => setTimeout(resolve, 450));
    assert.equal(
      JSON.parse(await fs.readFile(statePath, "utf8")).checkpointStatus,
      "saved",
    );
    assert.equal(
      await new Promise((resolve) => processHandle.once("close", resolve)),
      0,
    );
  } finally {
    processHandle.kill("SIGKILL");
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("pause waits for a checkpoint, then resumes the same training process", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-worker-pause-"));
  const worker = path.resolve("dist-electron/main/training-worker.js");
  const backend = path.resolve("tests/fixtures/fake-backend.mjs");
  const statePath = path.join(root, "state.json");
  const jobPath = path.join(root, "job.json");
  const now = new Date().toISOString();
  const id = "88888888-8888-4888-8888-888888888888";
  const state = {
    schemaVersion: 1,
    id,
    name: "pause-resume",
    status: "queued",
    phase: "preparing",
    progress: 0,
    indeterminate: true,
    message: "queued",
    createdAt: now,
    sessionDirectory: root,
    logPath: path.join(root, "training.log"),
    command: "node fake-backend.mjs --pause-checkpoint",
    request: { autoStop: false },
  };
  const job = {
    schemaVersion: 1,
    id,
    executable: process.execPath,
    args: [backend, "--pause-checkpoint", "--nested-trainer"],
    sessionDirectory: root,
    statePath,
    logPath: state.logPath,
    stopPath: path.join(root, "stop.request"),
    pausePath: path.join(root, "pause.request"),
    resumePath: path.join(root, "resume.request"),
    checkpointRequestPath: path.join(root, "checkpoint.request"),
    autoStopPath: path.join(root, "auto-stop.setting"),
    stage: "fine-tuning",
    iterations: 1,
    alignmentIterations: 1,
    createdAt: now,
  };
  await fs.writeFile(statePath, JSON.stringify(state));
  await fs.writeFile(jobPath, JSON.stringify(job));
  const processHandle = spawn(process.execPath, [worker, jobPath], {
    stdio: "ignore",
  });
  try {
    await waitFor(
      statePath,
      (value) => value.status === "running" && value.trainingPercent > 0,
    );
    await fs.writeFile(job.autoStopPath, "on\n");
    const enabled = await waitFor(
      statePath,
      (value) => value.autoStopEnabled === true,
    );
    assert.equal(enabled.request.autoStop, true);
    await fs.writeFile(job.autoStopPath, "off\n");
    const disabled = await waitFor(
      statePath,
      (value) => value.autoStopEnabled === false,
    );
    assert.equal(disabled.request.autoStop, false);
    await fs.writeFile(job.pausePath, "pause\n");
    const paused = await waitFor(
      statePath,
      (value) =>
        value.status === "paused" || value.checkpointStatus === "failed",
    );
    assert.equal(paused.status, "paused", paused.error);
    assert.equal(paused.checkpointStatus, "saved");
    assert.equal((await fs.stat(paused.checkpointPath)).size > 0, true);
    await new Promise((resolve) => setTimeout(resolve, 450));
    const stillPaused = JSON.parse(await fs.readFile(statePath, "utf8"));
    assert.equal(stillPaused.status, "paused");
    assert.equal(stillPaused.trainingPercent, paused.trainingPercent);
    const heartbeat = path.join(root, "nested-heartbeat.txt");
    const heartbeatSize = (await fs.stat(heartbeat)).size;
    await new Promise((resolve) => setTimeout(resolve, 320));
    assert.equal((await fs.stat(heartbeat)).size, heartbeatSize);
    await fs.writeFile(job.resumePath, "resume\n");
    await waitFor(
      statePath,
      (value) =>
        value.status === "running" &&
        value.trainingPercent > paused.trainingPercent,
    );
    const complete = await waitFor(
      statePath,
      (value) => value.status === "completed",
    );
    assert.equal(complete.progress, 100);
    const history = await fs.readFile(path.join(root, "metrics.csv"), "utf8");
    assert.match(history, /checkpoint/);
    assert.match(history, /,loss\n/);
  } finally {
    await fs.writeFile(job.stopPath, "stop\n").catch(() => undefined);
    await waitFor(
      statePath,
      (value) => ["stopped", "completed", "failed"].includes(value.status),
      5_000,
    ).catch(() => undefined);
    processHandle.kill("SIGKILL");
    await fs.rm(root, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 100,
    });
  }
});

test("Auto stop saves a checkpoint before ending a rising-loss trainer", async () => {
  const root = await fs.mkdtemp(
    path.join(os.tmpdir(), "osai-worker-auto-stop-"),
  );
  const worker = path.resolve("dist-electron/main/training-worker.js");
  const clock = path.resolve("tests/fixtures/fast-clock.cjs");
  const backend = path.resolve("tests/fixtures/fake-backend.mjs");
  const statePath = path.join(root, "state.json");
  const jobPath = path.join(root, "job.json");
  const id = "77777777-7777-4777-8777-777777777777";
  const now = new Date().toISOString();
  const state = {
    schemaVersion: 1,
    id,
    name: "auto-stop",
    status: "queued",
    phase: "preparing",
    progress: 0,
    indeterminate: true,
    autoStopEnabled: true,
    message: "queued",
    createdAt: now,
    sessionDirectory: root,
    logPath: path.join(root, "training.log"),
    command: "node fake-backend.mjs --auto-stop",
  };
  const job = {
    schemaVersion: 1,
    id,
    executable: process.execPath,
    args: [backend, "--auto-stop"],
    sessionDirectory: root,
    statePath,
    logPath: state.logPath,
    stopPath: path.join(root, "stop.request"),
    checkpointRequestPath: path.join(root, "checkpoint.request"),
    autoStopPath: path.join(root, "auto-stop.setting"),
    stage: "fine-tuning",
    iterations: 1,
    alignmentIterations: 1,
    createdAt: now,
  };
  await fs.writeFile(statePath, JSON.stringify(state));
  await fs.writeFile(jobPath, JSON.stringify(job));
  const handle = spawn(
    process.execPath,
    ["--require", clock, worker, jobPath],
    {
      stdio: "ignore",
    },
  );
  try {
    const stopped = await waitFor(
      statePath,
      (value) => value.status === "stopped",
    );
    assert.equal(stopped.checkpointStatus, "saved");
    assert.match(stopped.message, /Auto stop saved a checkpoint/);
    assert.equal((await fs.stat(stopped.checkpointPath)).size > 0, true);
  } finally {
    handle.kill("SIGKILL");
    await fs.rm(root, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 100,
    });
  }
});

test("MLX table iterations advance against the announced optimizer steps", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-worker-mlx-"));
  const worker = path.resolve("dist-electron/main/training-worker.js");
  const backend = path.resolve("tests/fixtures/fake-backend.mjs");
  const statePath = path.join(root, "state.json");
  const logPath = path.join(root, "training.log");
  const jobPath = path.join(root, "job.json");
  const now = new Date().toISOString();
  const state = {
    schemaVersion: 1,
    id: "66666666-6666-4666-8666-666666666666",
    name: "mlx-progress-test",
    status: "queued",
    phase: "preparing",
    progress: 0,
    indeterminate: true,
    message: "queued",
    createdAt: now,
    sessionDirectory: root,
    logPath,
    command: "node fake-backend.mjs --mlx-progress",
  };
  const job = {
    schemaVersion: 1,
    id: state.id,
    executable: process.execPath,
    args: [backend, "--mlx-progress"],
    sessionDirectory: root,
    statePath,
    logPath,
    stopPath: path.join(root, "stop.request"),
    stage: "fine-tuning",
    iterations: 1,
    alignmentIterations: 1,
    createdAt: now,
  };
  await fs.writeFile(statePath, JSON.stringify(state));
  await fs.writeFile(jobPath, JSON.stringify(job));
  const processHandle = spawn(process.execPath, [worker, jobPath], {
    stdio: "ignore",
  });
  try {
    const progressing = await waitFor(
      statePath,
      (value) => value.status === "running" && value.progress >= 25,
    );
    assert.equal(progressing.progress, 25);
    assert.match(
      progressing.message,
      /Fine-tuning: 3,219 of 15,011 \(21\.44%\)/,
    );
    await waitFor(statePath, (value) => value.status === "completed");
  } finally {
    processHandle.kill("SIGKILL");
    await fs.rm(root, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 100,
    });
  }
});

test("auto settings and memory retry remain visible in session status", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-worker-auto-"));
  const worker = path.resolve("dist-electron/main/training-worker.js");
  const backend = path.resolve("tests/fixtures/fake-backend.mjs");
  const statePath = path.join(root, "state.json");
  const logPath = path.join(root, "training.log");
  const jobPath = path.join(root, "job.json");
  const now = new Date().toISOString();
  const state = {
    schemaVersion: 1,
    id: "66666666-6666-4666-8666-666666666666",
    name: "auto-retry",
    status: "queued",
    phase: "preparing",
    progress: 0,
    indeterminate: true,
    message: "queued",
    createdAt: now,
    sessionDirectory: root,
    logPath,
    command: "node fake-backend.mjs --auto-retry",
  };
  const job = {
    schemaVersion: 1,
    id: state.id,
    executable: process.execPath,
    args: [backend, "--auto-retry"],
    sessionDirectory: root,
    statePath,
    logPath,
    stopPath: path.join(root, "stop.request"),
    stage: "fine-tuning",
    iterations: 1,
    alignmentIterations: 1,
    createdAt: now,
  };
  await fs.writeFile(statePath, JSON.stringify(state));
  await fs.writeFile(jobPath, JSON.stringify(job));
  const processHandle = spawn(process.execPath, [worker, jobPath], {
    stdio: "ignore",
  });
  try {
    const active = await waitFor(statePath, (value) =>
      Boolean(value.adjustment),
    );
    assert.match(active.autoSettingsSummary, /Context 512/);
    assert.match(active.adjustment, /Attempt 2.*1024 → 512/);
    assert.equal(active.indeterminate, true);
    await waitFor(statePath, (value) => value.status === "completed");
  } finally {
    processHandle.kill("SIGKILL");
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("a failed worker surfaces the backend error instead of only its exit code", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-worker-fail-"));
  const worker = path.resolve("dist-electron/main/training-worker.js");
  const backend = path.resolve("tests/fixtures/fake-backend.mjs");
  const statePath = path.join(root, "state.json");
  const logPath = path.join(root, "training.log");
  const jobPath = path.join(root, "job.json");
  const now = new Date().toISOString();
  const state = {
    schemaVersion: 1,
    id: "55555555-5555-4555-8555-555555555555",
    name: "failure-test",
    status: "queued",
    phase: "preparing",
    progress: 0,
    indeterminate: true,
    message: "queued",
    createdAt: now,
    sessionDirectory: root,
    logPath,
    command: "node fake-backend.mjs --fail",
  };
  const job = {
    schemaVersion: 1,
    id: state.id,
    executable: process.execPath,
    args: [backend, "--fail"],
    sessionDirectory: root,
    statePath,
    logPath,
    stopPath: path.join(root, "stop.request"),
    stage: "fine-tuning",
    iterations: 1,
    alignmentIterations: 1,
    createdAt: now,
  };
  await fs.writeFile(statePath, JSON.stringify(state));
  await fs.writeFile(jobPath, JSON.stringify(job));
  const processHandle = spawn(process.execPath, [worker, jobPath], {
    stdio: "ignore",
  });
  try {
    const failed = await waitFor(
      statePath,
      (value) => value.status === "failed",
    );
    assert.equal(failed.exitCode, 2);
    assert.equal(failed.error, "unsupported dataset schema at train.jsonl:1");
  } finally {
    processHandle.kill("SIGKILL");
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("a detached training worker stops only after an explicit stop request", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-worker-stop-"));
  const worker = path.resolve("dist-electron/main/training-worker.js");
  const backend = path.resolve("tests/fixtures/fake-backend.mjs");
  const statePath = path.join(root, "state.json");
  const jobPath = path.join(root, "job.json");
  const stopPath = path.join(root, "stop.request");
  const id = "22222222-2222-4222-8222-222222222222";
  const now = new Date().toISOString();
  const state = {
    schemaVersion: 1,
    id,
    name: "stop-test",
    status: "queued",
    phase: "preparing",
    progress: 0,
    indeterminate: true,
    message: "queued",
    createdAt: now,
    sessionDirectory: root,
    logPath: path.join(root, "training.log"),
    command: "node fake-backend.mjs --long",
  };
  const job = {
    schemaVersion: 1,
    id,
    executable: process.execPath,
    args: [backend, "--long"],
    sessionDirectory: root,
    statePath,
    logPath: state.logPath,
    stopPath,
    stage: "fine-tuning",
    iterations: 2,
    alignmentIterations: 1,
    createdAt: now,
  };
  await fs.writeFile(statePath, JSON.stringify(state));
  await fs.writeFile(jobPath, JSON.stringify(job));
  const processHandle = spawn(process.execPath, [worker, jobPath], {
    stdio: "ignore",
  });
  try {
    await waitFor(statePath, (value) => value.status === "running");
    await fs.writeFile(stopPath, "stop\n");
    const stopped = await waitFor(
      statePath,
      (value) => value.status === "stopped",
    );
    assert.equal(stopped.indeterminate, false);
    assert.match(stopped.message, /stopped by the user/i);
  } finally {
    processHandle.kill("SIGKILL");
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("a checkpoint-paused worker can be stopped cleanly", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-worker-pause-"));
  const worker = path.resolve("dist-electron/main/training-worker.js");
  const backend = path.resolve("tests/fixtures/fake-backend.mjs");
  const statePath = path.join(root, "state.json");
  const jobPath = path.join(root, "job.json");
  const stopPath = path.join(root, "stop.request");
  const pausePath = path.join(root, "pause.request");
  const resumePath = path.join(root, "resume.request");
  const id = "44444444-4444-4444-8444-444444444444";
  const now = new Date().toISOString();
  const state = {
    schemaVersion: 1,
    id,
    name: "pause-test",
    status: "queued",
    phase: "preparing",
    progress: 0,
    indeterminate: true,
    message: "queued",
    createdAt: now,
    sessionDirectory: root,
    logPath: path.join(root, "training.log"),
    command: "node fake-backend.mjs --long",
  };
  const job = {
    schemaVersion: 1,
    id,
    executable: process.execPath,
    args: [backend, "--pause-checkpoint"],
    sessionDirectory: root,
    statePath,
    logPath: state.logPath,
    stopPath,
    pausePath,
    resumePath,
    checkpointRequestPath: path.join(root, "checkpoint.request"),
    stage: "fine-tuning",
    iterations: 2,
    alignmentIterations: 1,
    createdAt: now,
  };
  await fs.writeFile(statePath, JSON.stringify(state));
  await fs.writeFile(jobPath, JSON.stringify(job));
  const processHandle = spawn(process.execPath, [worker, jobPath], {
    stdio: "ignore",
  });
  try {
    await waitFor(
      statePath,
      (value) => value.status === "running" && value.trainingPercent > 0,
    );
    await fs.writeFile(pausePath, "pause\n");
    const paused = await waitFor(
      statePath,
      (value) =>
        value.status === "paused" || value.checkpointStatus === "failed",
    );
    assert.equal(paused.status, "paused", paused.error);
    assert.equal(paused.indeterminate, false);
    assert.equal(paused.checkpointStatus, "saved");
    await fs.writeFile(stopPath, "stop\n");
    await waitFor(statePath, (value) => value.status === "stopped");
  } finally {
    await fs.writeFile(stopPath, "stop\n").catch(() => undefined);
    await waitFor(
      statePath,
      (value) => ["stopped", "completed", "failed"].includes(value.status),
      5_000,
    ).catch(() => undefined);
    processHandle.kill("SIGKILL");
    await fs.rm(root, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 100,
    });
  }
});
