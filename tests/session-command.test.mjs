import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  buildOsAiArgs,
  absolutizeDatasetMedia,
  formatTerminalOutput,
  prepareDatasetSelection,
  prepareSharedFineTuneData,
  restoreLegacyRequest,
  SessionService,
} from "../dist-electron/main/session-service.js";

const base = {
  sessionsRoot: "",
  modelSource: "official",
  tier: "small",
  customModelFolder: "",
  engine: "auto",
  accelerator: "auto",
  stage: "fine-tune-align",
  fineTuneData: "",
  alignmentData: "",
  reuseDataset: false,
  adapter: "",
  alignmentType: "grpo",
  optimizer: "auto",
  autoSettings: true,
  multiGpu: "auto",
  liveRollouts: true,
  sessionName: "My safe run",
  iterations: 2,
  alignmentIterations: 4,
  batchSize: null,
  gradientAccumulationSteps: null,
  gradientCheckpoint: true,
  maxSeqLength: null,
  learningRate: null,
  alignmentLearningRate: null,
  rank: null,
  scale: null,
  numLayers: null,
  dropout: null,
  imageWidth: null,
  imageHeight: null,
  videoFps: 2,
  videoMaxFrames: 32,
  assistantTokenId: null,
  seed: null,
  saveEvery: null,
  stepsPerReport: null,
  stepsPerEval: null,
  validationBatches: null,
  maskPrompt: true,
  targetModules: [],
  alignmentBeta: 0.1,
  alignmentGamma: 0.5,
  ppoClip: 0.2,
  rolloutMaxTokens: 32,
  rolloutsPerPrompt: 3,
  rolloutTemperature: 0.8,
  rolloutTopP: 0.95,
  rolloutSeed: 0,
  ggufBatchSize: null,
  ggufThreads: null,
  distributedWorkers: null,
  splitMode: "auto",
  tensorSplit: "",
  mainGpu: null,
  devices: "",
};

test("resolves relative media beside an individually selected dataset", () => {
  const result = absolutizeDatasetMedia(
    {
      image: "images/a.png",
      messages: [
        {
          role: "user",
          content: [
            { type: "video", video: "clips/a.mp4" },
            { type: "audio_url", audio_url: { path: "sound/a.wav" } },
          ],
        },
      ],
    },
    "/datasets/demo",
  );
  assert.equal(result.image, "/datasets/demo/images/a.png");
  assert.equal(
    result.messages[0].content[0].video,
    "/datasets/demo/clips/a.mp4",
  );
  assert.equal(
    result.messages[0].content[1].audio_url.path,
    "/datasets/demo/sound/a.wav",
  );
});

test("renders carriage-return terminal progress as one current line", () => {
  const output = formatTerminalOutput(
    "start\n\n\r[------] 0% config.json\r[#-----] 5% model.gguf\n\x1b[31mdone\x1b[0m\n",
  );
  assert.equal(output, "start\n\n[#-----] 5% model.gguf\ndone");
  assert.doesNotMatch(output, /config\.json|\x1b/);
});

test("restores settings from sessions created by earlier app builds", () => {
  const sessionDirectory = path.join(
    os.homedir(),
    "Library",
    "Application Support",
    "osai-app",
    "sessions",
    "older-run",
  );
  const job = {
    schemaVersion: 1,
    id: "33333333-3333-3333-3333-333333333333",
    executable: "osai",
    args: [
      "train",
      "--tier",
      "small",
      "--engine",
      "auto",
      "--accelerator",
      "auto",
      "--stage",
      "fine-tuning",
      "--optimizer",
      "auto",
      "--multi-gpu",
      "auto",
      "--auto-settings",
      "--data",
      "/datasets/dolly.jsonl",
      "--iterations",
      "3",
      "--session-name",
      "dolly-run",
    ],
    sessionDirectory,
    statePath: path.join(sessionDirectory, "state.json"),
    logPath: path.join(sessionDirectory, "training.log"),
    stopPath: path.join(sessionDirectory, "stop.request"),
    stage: "fine-tuning",
    iterations: 3,
    alignmentIterations: 10,
    createdAt: "2026-01-01T00:00:00.000Z",
  };
  const restored = restoreLegacyRequest(job, {
    schemaVersion: 1,
    id: job.id,
    name: "dolly-run",
    status: "completed",
    phase: "complete",
    progress: 100,
    indeterminate: false,
    message: "Complete",
    createdAt: job.createdAt,
    sessionDirectory,
    logPath: job.logPath,
    command: "osai train",
  });
  assert.equal(restored.sessionsRoot, path.dirname(sessionDirectory));
  assert.equal(restored.sessionName, "dolly-run");
  assert.equal(restored.fineTuneData, "/datasets/dolly.jsonl");
  assert.equal(restored.iterations, 3);
  assert.equal(restored.autoSettings, true);
});

test("keeps current, custom, and legacy session locations discoverable", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-session-roots-"));
  const current = path.join(root, "current");
  const legacy = path.join(root, "legacy");
  await fs.mkdir(current);
  await fs.mkdir(legacy);
  const makeState = async (sessionsRoot, id, name) => {
    const directory = path.join(sessionsRoot, name);
    await fs.mkdir(directory);
    await fs.writeFile(
      path.join(directory, "state.json"),
      JSON.stringify({
        schemaVersion: 1,
        id,
        name,
        status: "completed",
        phase: "complete",
        progress: 100,
        indeterminate: false,
        message: "Complete",
        createdAt: "2026-01-01T00:00:00.000Z",
        sessionDirectory: directory,
        logPath: path.join(directory, "training.log"),
        command: "osai train",
        request: { ...base, sessionsRoot },
      }),
    );
  };
  await makeState(current, "11111111-1111-1111-1111-111111111111", "new");
  await makeState(legacy, "22222222-2222-2222-2222-222222222222", "old");
  const service = new SessionService(legacy, "unused-worker", async () => ({
    version: 1,
    theme: "dark",
    backendExecutable: "",
    autoUpdateEnabled: false,
    sessionsRoot: current,
    sessionRoots: [current],
  }));
  try {
    assert.equal(await service.root(), current);
    const sessions = await service.list();
    assert.deepEqual(sessions.map((session) => session.name).sort(), [
      "new",
      "old",
    ]);
    assert.equal(
      sessions.find((session) => session.name === "new").request.tier,
      "small",
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("recovers visible MLX progress from an active session log", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-progress-log-"));
  const directory = path.join(root, "active");
  const id = "77777777-7777-4777-8777-777777777777";
  const logPath = path.join(directory, "training.log");
  await fs.mkdir(directory);
  await fs.writeFile(
    logPath,
    [
      "osai: training plan examples=15011 epochs=1 batch=1 steps=15011 optimizer_updates=15011",
      "iter   train_loss     tok/s     tokens",
      "3219    1.845 ▼    25    101.0k",
    ].join("\n"),
  );
  await fs.writeFile(
    path.join(directory, "state.json"),
    JSON.stringify({
      schemaVersion: 1,
      id,
      name: "active",
      status: "running",
      phase: "preparing",
      progress: 1,
      indeterminate: true,
      message: "Checking the model and local training backend",
      createdAt: new Date().toISOString(),
      startedAt: new Date().toISOString(),
      workerPid: process.pid,
      sessionDirectory: directory,
      logPath,
      command: "osai train",
      request: {
        ...base,
        sessionsRoot: root,
        stage: "fine-tuning",
        iterations: 1,
      },
    }),
  );
  const service = new SessionService(root, "unused-worker", async () => ({
    version: 1,
    theme: "dark",
    backendExecutable: "",
    autoUpdateEnabled: false,
    sessionsRoot: root,
    sessionRoots: [],
  }));
  try {
    const [session] = await service.list();
    assert.equal(session.progress, 25);
    assert.equal(session.phase, "fine-tuning");
    assert.equal(session.indeterminate, false);
    assert.equal(session.message, "Fine-tuning update 3219 of 15011");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("moves finished sessions to Trash and protects active sessions", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-delete-session-"));
  const completedDirectory = path.join(root, "completed");
  const activeDirectory = path.join(root, "active");
  const completedId = "33333333-3333-4333-8333-333333333333";
  const activeId = "44444444-4444-4444-8444-444444444444";
  await fs.mkdir(completedDirectory);
  await fs.mkdir(activeDirectory);
  const state = (id, name, status, directory) => ({
    schemaVersion: 1,
    id,
    name,
    status,
    phase: status === "completed" ? "complete" : "fine-tuning",
    progress: status === "completed" ? 100 : 20,
    indeterminate: false,
    message: status,
    createdAt: new Date().toISOString(),
    // Deliberately forged: list() must replace this with the scanned folder.
    sessionDirectory: path.join(root, "not-the-session"),
    logPath: path.join(root, "not-the-session", "training.log"),
    command: "osai train",
    request: { ...base, sessionsRoot: root },
  });
  await fs.writeFile(
    path.join(completedDirectory, "state.json"),
    JSON.stringify(state(completedId, "completed", "completed")),
  );
  await fs.writeFile(
    path.join(activeDirectory, "state.json"),
    JSON.stringify(state(activeId, "active", "running")),
  );
  const service = new SessionService(root, "unused-worker", async () => ({
    version: 1,
    theme: "dark",
    backendExecutable: "",
    autoUpdateEnabled: false,
    sessionsRoot: root,
    sessionRoots: [],
  }));
  const trashed = [];
  try {
    await service.remove(completedId, async (directory) => {
      trashed.push(directory);
    });
    assert.deepEqual(trashed, [completedDirectory]);
    await assert.rejects(
      service.remove(activeId, async () => undefined),
      /Stop this training session before deleting it/,
    );
    await assert.rejects(
      service.restart(activeId, base, async () => undefined),
      /Stop this training session before restarting it/,
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("restart clears the previous pipeline before launching saved settings", async () => {
  const root = await fs.mkdtemp(
    path.join(os.tmpdir(), "osai-restart-session-"),
  );
  const directory = path.join(root, "failed");
  const dataset = path.join(root, "dataset");
  const id = "55555555-5555-4555-8555-555555555555";
  const request = {
    ...base,
    sessionsRoot: root,
    stage: "fine-tuning",
    fineTuneData: dataset,
    alignmentData: "",
  };
  await fs.mkdir(directory);
  await fs.mkdir(dataset);
  await fs.writeFile(
    path.join(directory, "state.json"),
    JSON.stringify({
      schemaVersion: 1,
      id,
      name: "failed",
      status: "failed",
      phase: "preparing",
      progress: 1,
      indeterminate: false,
      message: "failed",
      error: "stale output lock",
      createdAt: new Date().toISOString(),
      sessionDirectory: directory,
      logPath: path.join(directory, "training.log"),
      command: "osai train",
      request,
    }),
  );
  const service = new SessionService(root, "unused-worker", async () => ({
    version: 1,
    theme: "dark",
    backendExecutable: "",
    autoUpdateEnabled: false,
    sessionsRoot: root,
    sessionRoots: [],
  }));
  const events = [];
  service.start = async (input) => {
    events.push(["start", input]);
    return { id: "replacement" };
  };
  try {
    const restarted = await service.restart(id, request, async (target) => {
      events.push(["trash", target]);
    });
    assert.equal(restarted.id, "replacement");
    assert.equal(events[0][0], "trash");
    assert.equal(events[0][1], directory);
    assert.equal(events[1][0], "start");
    assert.deepEqual(events[1][1], request);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("prepares individual JSON, JSONL, and NDJSON dataset files for the CLI", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-dataset-file-"));
  const json = path.join(root, "fine-tune.json");
  const jsonl = path.join(root, "alignment.jsonl");
  const ndjson = path.join(root, "chat.ndjson");
  await fs.writeFile(
    json,
    JSON.stringify([
      { prompt: "One", completion: "First" },
      { prompt: "Two", completion: "Second" },
    ]),
  );
  await fs.writeFile(
    jsonl,
    `${JSON.stringify({ prompt: "One", chosen: "Yes", rejected: "No" })}\n`,
  );
  await fs.writeFile(
    ndjson,
    `${JSON.stringify({
      messages: [
        { role: "user", content: "Hi" },
        { role: "assistant", content: "Hello" },
      ],
    })}\n`,
  );
  try {
    const fine = await prepareDatasetSelection(
      json,
      path.join(root, "prepared-fine"),
      "Fine-tuning dataset",
    );
    const align = await prepareDatasetSelection(
      jsonl,
      path.join(root, "prepared-align"),
      "Alignment dataset",
    );
    const chat = await prepareDatasetSelection(
      ndjson,
      path.join(root, "prepared-chat"),
      "Fine-tuning dataset",
    );
    assert.equal(
      await fs.readFile(path.join(fine, "train.jsonl"), "utf8"),
      '{"prompt":"One","completion":"First"}\n{"prompt":"Two","completion":"Second"}\n',
    );
    assert.equal(
      await fs.readFile(path.join(align, "train.jsonl"), "utf8"),
      '{"prompt":"One","chosen":"Yes","rejected":"No"}\n',
    );
    assert.match(
      await fs.readFile(path.join(chat, "train.jsonl"), "utf8"),
      /"messages"/,
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("uses an existing dataset directory without copying it", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-dataset-dir-"));
  try {
    assert.equal(
      await prepareDatasetSelection(
        root,
        path.join(root, "unused"),
        "Fine-tuning dataset",
      ),
      root,
    );
    await assert.rejects(fs.stat(path.join(root, "unused")));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("builds a non-interactive combined osAi command with local rollouts", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-command-"));
  const fine = path.join(root, "fine");
  const align = path.join(root, "align");
  await fs.mkdir(fine);
  await fs.mkdir(align);
  try {
    const { args, sessionName } = await buildOsAiArgs(
      { ...base, fineTuneData: fine, alignmentData: align },
      path.join(root, "sessions"),
      path.join(root, "models"),
    );
    assert.deepEqual(args.slice(0, 3), ["train", "--tier", "small"]);
    assert.equal(args.includes("--auto-settings"), true);
    assert.equal(args.includes("--live-rollouts"), true);
    assert.equal(args[args.indexOf("--alignment-type") + 1], "grpo");
    assert.equal(args[args.indexOf("--data") + 1], fine);
    assert.equal(args[args.indexOf("--alignment-data") + 1], align);
    assert.equal(
      args[args.indexOf("--bundled-root") + 1],
      path.join(root, "models"),
    );
    assert.equal(sessionName, "My-safe-run");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("passes organized manual training, rollout, and runtime controls", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-advanced-"));
  const fine = path.join(root, "fine");
  const align = path.join(root, "align");
  await fs.mkdir(fine);
  await fs.mkdir(align);
  try {
    const { args } = await buildOsAiArgs(
      {
        ...base,
        fineTuneData: fine,
        alignmentData: align,
        autoSettings: false,
        batchSize: 3,
        gradientAccumulationSteps: 4,
        gradientCheckpoint: false,
        maxSeqLength: 512,
        learningRate: 0.00002,
        alignmentLearningRate: 0.00001,
        rank: 8,
        scale: 16,
        numLayers: 6,
        dropout: 0.1,
        seed: 42,
        saveEvery: 20,
        stepsPerReport: 2,
        stepsPerEval: 10,
        validationBatches: -1,
        maskPrompt: false,
        targetModules: ["self_attn.q_proj", "mlp.down_proj"],
        alignmentBeta: 0.2,
        alignmentGamma: 0.7,
        ppoClip: 0.15,
        rolloutTemperature: 0.6,
        rolloutTopP: 0.9,
        rolloutSeed: 7,
        ggufBatchSize: 16,
        ggufThreads: 6,
        distributedWorkers: 2,
        splitMode: "row",
        tensorSplit: "3, 1",
        mainGpu: 1,
        devices: "CUDA0, CUDA1",
      },
      path.join(root, "sessions"),
    );
    const value = (flag) => args[args.indexOf(flag) + 1];
    assert.equal(value("--batch-size"), "3");
    assert.equal(value("--epochs"), "2");
    assert.equal(args.includes("--iterations"), false);
    assert.equal(value("--gradient-accumulation-steps"), "4");
    assert.equal(args.includes("--no-gradient-checkpointing"), true);
    assert.equal(value("--max-seq-length"), "512");
    assert.equal(value("--learning-rate"), "0.00002");
    assert.equal(value("--alignment-learning-rate"), "0.00001");
    assert.equal(value("--rank"), "8");
    assert.equal(value("--scale"), "16");
    assert.equal(value("--num-layers"), "6");
    assert.equal(value("--dropout"), "0.1");
    assert.equal(value("--seed"), "42");
    assert.equal(value("--save-every"), "20");
    assert.equal(value("--steps-per-report"), "2");
    assert.equal(value("--steps-per-eval"), "10");
    assert.equal(value("--val-batches"), "-1");
    assert.equal(args.includes("--no-mask-prompt"), true);
    assert.deepEqual(
      args.filter((value, index) => args[index - 1] === "--target-module"),
      ["self_attn.q_proj", "mlp.down_proj"],
    );
    assert.equal(value("--alignment-beta"), "0.2");
    assert.equal(value("--alignment-gamma"), "0.7");
    assert.equal(value("--ppo-clip"), "0.15");
    assert.equal(value("--rollout-temperature"), "0.6");
    assert.equal(value("--rollout-top-p"), "0.9");
    assert.equal(value("--rollout-seed"), "7");
    assert.equal(value("--gguf-batch-size"), "16");
    assert.equal(value("--gguf-threads"), "6");
    assert.equal(value("--distributed-workers"), "2");
    assert.equal(value("--split-mode"), "row");
    assert.equal(value("--tensor-split"), "3,1");
    assert.equal(value("--main-gpu"), "1");
    assert.deepEqual(
      args.filter((item, index) => args[index - 1] === "--device"),
      ["CUDA0", "CUDA1"],
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("lets automatic hardware fitting own memory-sensitive controls", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-auto-"));
  const fine = path.join(root, "fine");
  await fs.mkdir(fine);
  try {
    const { args, sessionName } = await buildOsAiArgs(
      {
        ...base,
        stage: "fine-tuning",
        fineTuneData: fine,
        sessionName: "",
        batchSize: 99,
        maxSeqLength: 999,
        rank: 99,
        numLayers: 99,
        ggufBatchSize: 99,
        ggufThreads: 99,
        targetModules: ["self_attn.q_proj"],
      },
      path.join(root, "sessions"),
    );
    for (const flag of [
      "--batch-size",
      "--max-seq-length",
      "--rank",
      "--num-layers",
      "--gguf-batch-size",
      "--gguf-threads",
      "--target-module",
      "--session-name",
    ])
      assert.equal(args.includes(flag), false, `${flag} should be omitted`);
    assert.equal(sessionName, "small-fine-tuning");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("rejects invalid advanced sampling controls", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-invalid-"));
  const fine = path.join(root, "fine");
  const align = path.join(root, "align");
  await fs.mkdir(fine);
  await fs.mkdir(align);
  try {
    await assert.rejects(
      () =>
        buildOsAiArgs(
          {
            ...base,
            fineTuneData: fine,
            alignmentData: align,
            rolloutTopP: 1.5,
          },
          path.join(root, "sessions"),
        ),
      /Rollout top-p/,
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("rejects missing datasets before a worker is launched", async () => {
  await assert.rejects(
    () =>
      buildOsAiArgs(
        {
          ...base,
          fineTuneData: "/missing/fine",
          alignmentData: "/missing/align",
        },
        "/tmp/sessions",
      ),
    /does not exist/,
  );
});

test("prepares one preference dataset safely for both pipeline stages", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-shared-data-"));
  const source = path.join(root, "source");
  const prepared = path.join(root, "prepared");
  await fs.mkdir(source);
  await fs.writeFile(
    path.join(source, "train.jsonl"),
    `${JSON.stringify({ prompt: "Name a colour", chosen: "Blue", rejected: "Banana" })}\n`,
  );
  try {
    await prepareSharedFineTuneData(source, prepared);
    assert.deepEqual(
      JSON.parse(await fs.readFile(path.join(prepared, "train.jsonl"), "utf8")),
      { prompt: "Name a colour", chosen: "Blue", rejected: "Banana" },
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
