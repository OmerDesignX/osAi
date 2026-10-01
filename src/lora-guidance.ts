export type LearningPace =
  "very-gentle" | "gentle" | "balanced" | "brisk" | "fast" | "rapid";

export type LearningRateOption = {
  pace: LearningPace;
  label: string;
  rate: number;
};

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function learningRateOptions(input: {
  datasetBytes: number | null;
  modelBytes: number;
  memoryBytes: number;
  batchSize: number;
  rank: number;
  epochs: number;
}): LearningRateOption[] {
  // A conservative backend-default anchor, adjusted for update exposure and
  // adapter capacity. Learning rate itself does not control memory use.
  const datasetFactor =
    input.datasetBytes === null
      ? 1
      : clamp(
          Math.pow((64 * MIB) / Math.max(input.datasetBytes, MIB), 0.12),
          0.65,
          1.35,
        );
  const modelFactor = clamp(
    Math.pow((3 * GIB) / Math.max(input.modelBytes, GIB), 0.1),
    0.8,
    1.15,
  );
  const batchFactor = clamp(
    Math.sqrt(Math.max(input.batchSize, 1) / 2),
    0.7,
    1.4,
  );
  const rankFactor = clamp(Math.sqrt(8 / Math.max(input.rank, 1)), 0.8, 1.25);
  const epochFactor = clamp(1 / Math.sqrt(Math.max(input.epochs, 1)), 0.6, 1);
  const memoryFactor = clamp(
    Math.pow(Math.max(input.memoryBytes, 4 * GIB) / (16 * GIB), 0.04),
    0.9,
    1.1,
  );
  const center =
    1e-5 *
    datasetFactor *
    modelFactor *
    batchFactor *
    rankFactor *
    epochFactor *
    memoryFactor;
  const paces: Array<[LearningPace, string, number]> = [
    ["very-gentle", "Very gentle", 0.25],
    ["gentle", "Gentle", 0.5],
    ["balanced", "Balanced", 1],
    ["brisk", "Brisk", 1.5],
    ["fast", "Fast", 2.25],
    ["rapid", "Rapid", 3],
  ];
  return paces.map(([pace, label, multiplier]) => ({
    pace,
    label,
    rate: Number(clamp(center * multiplier, 1e-6, 8e-5).toPrecision(3)),
  }));
}

export function calibratedLearningRateOptions(
  measuredRate: number,
): LearningRateOption[] {
  const choices: Array<[LearningPace, string, number]> = [
    ["very-gentle", "Very gentle", 0.25],
    ["gentle", "Gentle", 0.5],
    ["balanced", "Measured fit", 1],
  ];
  return choices.map(([pace, label, multiplier]) => ({
    pace,
    label,
    rate: Number((measuredRate * multiplier).toPrecision(3)),
  }));
}

export function fittedChoices(maximum: number, choices: number[]) {
  return [
    ...new Set([1, ...choices.filter((value) => value <= maximum), maximum]),
  ]
    .filter((value) => value >= 1)
    .sort((left, right) => left - right);
}
