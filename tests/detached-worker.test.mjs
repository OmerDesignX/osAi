import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

async function waitFor(file, predicate, timeout = 8_000) {
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

test("a detached training worker pauses, resumes, and then stops", async () => {
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
    args: [backend, "--long"],
    sessionDirectory: root,
    statePath,
    logPath: state.logPath,
    stopPath,
    pausePath,
    resumePath,
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
    await fs.writeFile(pausePath, "pause\n");
    const paused = await waitFor(
      statePath,
      (value) => value.status === "paused",
    );
    assert.equal(paused.indeterminate, false);
    await fs.writeFile(resumePath, "resume\n");
    const resumed = await waitFor(
      statePath,
      (value) => value.status === "running" && /resumed/i.test(value.message),
    );
    assert.match(resumed.message, /resumed/i);
    await fs.writeFile(stopPath, "stop\n");
    await waitFor(statePath, (value) => value.status === "stopped");
  } finally {
    processHandle.kill("SIGKILL");
    await fs.rm(root, { recursive: true, force: true });
  }
});
