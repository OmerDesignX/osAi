import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  buildOsAiArgs,
  prepareSharedFineTuneData,
} from "../dist-electron/main/session-service.js";

const base = {
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
    );
    assert.deepEqual(args.slice(0, 3), ["train", "--tier", "small"]);
    assert.equal(args.includes("--auto-settings"), true);
    assert.equal(args.includes("--live-rollouts"), true);
    assert.equal(args[args.indexOf("--alignment-type") + 1], "grpo");
    assert.equal(args[args.indexOf("--data") + 1], fine);
    assert.equal(args[args.indexOf("--alignment-data") + 1], align);
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
      { prompt: "Name a colour", completion: "Blue" },
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
