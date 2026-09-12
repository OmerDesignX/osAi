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
