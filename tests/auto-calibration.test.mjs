import assert from "node:assert/strict";
import test from "node:test";
import {
  autoCalibrationArgs,
  validateAutoCalibrationResult,
} from "../dist-electron/main/auto-calibration.js";
import { calibrationKey } from "../dist-electron/calibration-key.js";

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
  assert.equal(args.includes("--no-full-content-context"), true);
  assert.equal(
    autoCalibrationArgs({ ...request, fullContentContext: true }).includes(
      "--full-content-context",
    ),
    true,
  );
});

for (const count of [3, 4, 8]) {
  test(`calibration passes all ${count} GPUs and split weights in order`, () => {
    const devices = Array.from(
      { length: count },
      (_, index) => `Vulkan${index}`,
    );
    const args = autoCalibrationArgs({
      ...request,
      accelerator: "vulkan",
      devices: devices.join(", "),
      splitMode: "layer",
      tensorSplit: devices.map(() => 1).join(","),
      mainGpu: count - 1,
    });
    assert.deepEqual(
      args.filter((_, index) => args[index - 1] === "--device"),
      devices,
    );
    assert.equal(
      args[args.indexOf("--tensor-split") + 1],
      devices.map(() => 1).join(","),
    );
    assert.equal(args[args.indexOf("--main-gpu") + 1], String(count - 1));
  });
}

test("calibration refuses to count the same GPU twice", () => {
  assert.throws(
    () => autoCalibrationArgs({ ...request, devices: "CUDA0,CUDA1,CUDA1" }),
    /distinct/,
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

test("a small verified pilot decline is ready for training", () => {
  const pilot = {
    settings: { max_seq_length: 1024 },
    devices: ["CUDA0", "CUDA1"],
    device_speeds: [2.1, 1],
    learning_rate: 1e-5,
    first_loss: 3.703472,
    last_loss: 3.701817,
    improvement_percent: 0.0447,
  };
  assert.equal(validateAutoCalibrationResult(pilot), pilot);
  assert.equal(
    validateAutoCalibrationResult({ ...pilot, required_context: 1000 })
      .required_context,
    1000,
  );
  for (const invalid of [
    { ...pilot, last_loss: pilot.first_loss },
    { ...pilot, improvement_percent: 0 },
    { ...pilot, learning_rate: 0 },
    { ...pilot, settings: { max_seq_length: 0 } },
    { ...pilot, required_context: 2048 },
    { ...pilot, device_speeds: [1] },
    { ...pilot, device_speeds: [1, Number.NaN] },
  ])
    assert.throws(
      () => validateAutoCalibrationResult(invalid),
      /invalid calibration measurements/,
    );
});

test("background inventory refresh does not invalidate the approved request", () => {
  const key = calibrationKey(request);
  assert.equal(
    calibrationKey({
      ...request,
      deviceSignature: "refreshed hardware inventory",
      datasetSignature: "refreshed dataset summary",
    }),
    key,
  );
  assert.equal(calibrationKey({ ...request, learningRate: 5e-6 }), key);
});
