import assert from "node:assert/strict";
import test from "node:test";
import { fittedChoices, learningRateOptions } from "../src/lora-guidance.ts";

const baseline = {
  datasetBytes: 64 * 1024 ** 2,
  modelBytes: 3 * 1024 ** 3,
  memoryBytes: 16 * 1024 ** 3,
  batchSize: 2,
  rank: 8,
  epochs: 1,
};

test("learning rate choices remain ordered and respond to data, model and fitted hardware", () => {
  const options = learningRateOptions(baseline);
  assert.equal(options.length, 6);
  assert.deepEqual(
    options.map(({ pace }) => pace),
    ["very-gentle", "gentle", "balanced", "brisk", "fast", "rapid"],
  );
  assert.ok(
    options.every(
      ({ rate }, index) =>
        rate >= 1e-6 &&
        rate <= 8e-5 &&
        (index === 0 || rate > options[index - 1].rate),
    ),
  );
  assert.ok(
    learningRateOptions({ ...baseline, datasetBytes: 2 * 1024 ** 3 })[2].rate <
      options[2].rate,
  );
  assert.ok(
    learningRateOptions({ ...baseline, modelBytes: 8 * 1024 ** 3 })[2].rate <
      options[2].rate,
  );
  assert.ok(
    learningRateOptions({
      ...baseline,
      batchSize: 4,
      memoryBytes: 32 * 1024 ** 3,
    })[2].rate > options[2].rate,
  );
  assert.ok(
    learningRateOptions({ ...baseline, epochs: 5 })[2].rate < options[2].rate,
  );
});

test("manual batch and rank choices never exceed the fitted limit", () => {
  assert.deepEqual(fittedChoices(4, [1, 2, 4, 8, 16]), [1, 2, 4]);
  assert.deepEqual(fittedChoices(3, [1, 2, 4]), [1, 2, 3]);
});
