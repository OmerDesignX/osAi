import assert from "node:assert/strict";
import test from "node:test";
import {
  FineTuneProgressParser,
  phaseProgress,
  recoverFineTuneProgress,
} from "../dist-electron/main/training-progress.js";

test("parses MLX table rows using the CLI training plan", () => {
  const parser = new FineTuneProgressParser(1);
  assert.deepEqual(
    parser.consume(
      "osai: training plan examples=15011 epochs=1 batch=1 steps=15011 optimizer_updates=15011",
    ),
    { completed: 0, total: 15011 },
  );
  assert.equal(parser.consume("iter   train_loss     tok/s     tokens"), null);
  assert.deepEqual(
    parser.consume(
      "\u001b[38;5;244m3219\u001b[0m    \u001b[1;32m1.845 ▼\u001b[0m    25    101.0k",
    ),
    { completed: 3219, total: 15011 },
  );
  assert.equal(
    Math.round(phaseProgress(3219, 15011, "fine-tuning", "fine-tuning")),
    25,
  );
});

test("recovers the latest MLX iteration from a saved log", () => {
  const output = [
    "osai: training plan examples=15011 epochs=1 batch=1 steps=15011 optimizer_updates=15011",
    "iter   train_loss     tok/s     tokens",
    "1    3.681 ▼    4    0.0k",
    "3219    1.845 ▼    25    101.0k",
  ].join("\n");
  assert.deepEqual(recoverFineTuneProgress(output), {
    completed: 3219,
    total: 15011,
  });
});

test("tracks llama.cpp fine-tuning data progress", () => {
  const parser = new FineTuneProgressParser(1);
  assert.deepEqual(
    parser.consume("train: [#####] data=0000002/0000016 loss=2.7"),
    { completed: 2, total: 16 },
  );
});

test("uses native data progress even when an epoch appears on the same row", () => {
  const parser = new FineTuneProgressParser(1);
  assert.deepEqual(
    parser.consume("train: epoch 0 data=43199/487398400 loss=0.52±0.01"),
    { completed: 43199, total: 487398400 },
  );
});

test("combines labelled multi-GPU worker progress", () => {
  const parser = new FineTuneProgressParser(1);
  assert.equal(
    parser.consume("osai: data-parallel LoRA training on CUDA0, CUDA1"),
    null,
  );
  assert.deepEqual(parser.consume("[CUDA0] train: data=2/16 loss=2.7"), {
    completed: 2,
    total: 32,
  });
  assert.deepEqual(parser.consume("[CUDA1] train: data=3/16 loss=2.6"), {
    completed: 5,
    total: 32,
  });
  assert.deepEqual(parser.consume("[CUDA0] train: data=16/16 loss=2.5"), {
    completed: 19,
    total: 32,
  });
  assert.deepEqual(parser.consume("[CUDA1] train: data=16/16 loss=2.4"), {
    completed: 32,
    total: 32,
  });
});
