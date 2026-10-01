import assert from "node:assert/strict";
import test from "node:test";
import {
  autoCalibrationArgs,
  calibrationKey,
} from "../dist-electron/main/auto-calibration.js";

const request = {
  modelSource: "official",
  modelVersion: "v2",
  tier: "small",
  customModelFolder: "",
  engine: "auto",
  accelerator: "cuda",
  multiGpu: "on",
  devices: "CUDA0, CUDA1",
  fineTuneData: "C:\\Users\\oa\\dataTrain\\openThoughts\\train.jsonl",
  fullContentContext: false,
  optimizer: "auto",
  iterations: 1,
  scale: null,
  dropout: null,
  seed: null,
  gradientAccumulationSteps: null,
  gradientCheckpoint: true,
  maskPrompt: true,
  splitMode: "auto",
  tensorSplit: "",
  mainGpu: null,
  distributedWorkers: null,
};

test("calibration uses the selected dataset, devices and context mode", () => {
  const args = autoCalibrationArgs(request);
  assert.equal(args[0], "calibrate");
  assert.deepEqual(
    args.slice(args.indexOf("--data"), args.indexOf("--optimizer")),
    ["--data", request.fineTuneData],
  );
  assert.equal(args.filter((value) => value === "--device").length, 2);
  assert.equal(args.includes("--full-content-context"), false);
  assert.equal(
    autoCalibrationArgs({ ...request, fullContentContext: true }).includes(
      "--full-content-context",
    ),
    true,
  );
});

test("model, data, optimizer and epoch changes invalidate calibration", () => {
  for (const change of [
    { fineTuneData: "C:\\different.jsonl" },
    { modelVersion: "v1" },
    { optimizer: "adamw" },
    { iterations: 2 },
    { devices: "CUDA0" },
    { splitMode: "tensor" },
    { gradientAccumulationSteps: 4 },
  ])
    assert.notEqual(
      calibrationKey(request),
      calibrationKey({ ...request, ...change }),
    );
});

test("calibration pilots advanced runtime settings used by training", () => {
  const args = autoCalibrationArgs({
    ...request,
    scale: 8,
    dropout: 0.1,
    seed: 17,
    gradientAccumulationSteps: 2,
    gradientCheckpoint: false,
    maskPrompt: false,
    splitMode: "tensor",
    tensorSplit: "1,1",
    mainGpu: 1,
    distributedWorkers: 2,
  });
  for (const [flag, value] of [
    ["--scale", "8"],
    ["--dropout", "0.1"],
    ["--seed", "17"],
    ["--grad-accumulation-steps", "2"],
    ["--split-mode", "tensor"],
    ["--tensor-split", "1,1"],
    ["--main-gpu", "1"],
    ["--distributed-workers", "2"],
  ])
    assert.equal(args[args.indexOf(flag) + 1], value);
  assert.ok(args.includes("--no-gradient-checkpointing"));
  assert.ok(args.includes("--no-mask-prompt"));
});
