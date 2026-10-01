import assert from "node:assert/strict";
import test from "node:test";
import { LossRiseDetector } from "../dist-electron/main/auto-stop.js";

function point(device, loss, percent) {
  return {
    time: "",
    percent,
    device,
    loss,
    lossUncertainty: null,
    accuracy: null,
    accuracyUncertainty: null,
    event: "loss",
  };
}

test("Auto stop detects sustained rising loss even when a large dataset is under 1% complete", () => {
  const detector = new LossRiseDetector();
  detector.setEnabled(true);
  let result = null;
  for (let index = 0; index < 48; index += 1) {
    const loss = [1, 1.02, 1.04, 1.07][Math.floor(index / 12)];
    const at = 1_000 + index * 7_000;
    const percent = 0.01 + index / 10_000;
    result = detector.observe(point("CUDA0", loss, percent), at);
    assert.equal(result, null);
    result = detector.observe(point("CUDA1", loss, percent), at + 1);
  }
  assert.match(result, /average loss rose/);
});

test("Auto stop waits for five minutes of recent measurements", () => {
  const detector = new LossRiseDetector();
  detector.setEnabled(true);
  for (let index = 0; index < 48; index += 1) {
    const loss = [1, 1.02, 1.04, 1.07][Math.floor(index / 12)];
    assert.equal(
      detector.observe(
        point("CUDA0", loss, 0.01 + index / 10_000),
        1_000 + index * 6_000,
      ),
      null,
    );
  }
});

test("Auto stop ignores noisy or premature loss and resets when toggled", () => {
  const detector = new LossRiseDetector();
  detector.setEnabled(true);
  for (let index = 0; index < 120; index += 1) {
    const at = 1_000 + index * 7_000;
    const percent = 12 + index / 10;
    const noise = index % 2 === 0 ? 0.03 : -0.03;
    assert.equal(
      detector.observe(point("Trainer", 1 + noise, percent), at),
      null,
    );
  }
  detector.setEnabled(false);
  assert.equal(detector.observe(point("Trainer", 2, 50), 900_000), null);
  detector.setEnabled(true);
  assert.equal(detector.observe(point("Trainer", 2, 50), 900_000), null);
});
