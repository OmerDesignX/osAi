import assert from "node:assert/strict";
import test from "node:test";
import {
  lossMetric,
  metricRow,
  parseMetricRow,
} from "../dist-electron/main/training-metrics.js";

test("records Unicode loss statistics and checkpoint-ready CSV rows", () => {
  const metric = lossMetric(
    "[CUDA1] train: data=43/100 loss=0.51406±0.06031 acc=84.38±1.40%",
    43,
  );
  assert.equal(metric.device, "CUDA1");
  assert.equal(metric.loss, 0.51406);
  assert.equal(metric.lossUncertainty, 0.06031);
  assert.equal(metric.accuracy, 84.38);
  assert.equal(metric.accuracyUncertainty, 1.4);
  assert.deepEqual(parseMetricRow(metricRow(metric).trim()), metric);
});

test("reads older replacement glyphs in a training log", () => {
  const metric = lossMetric("train: loss=0.5�0.1 acc=80�2%", 5);
  assert.equal(metric.lossUncertainty, 0.1);
  assert.equal(metric.accuracyUncertainty, 2);
});
