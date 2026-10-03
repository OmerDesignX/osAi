export type TrainingMetric = {
  time: string;
  percent: number;
  step?: number | null;
  device: string;
  loss: number | null;
  lossUncertainty: number | null;
  accuracy: number | null;
  accuracyUncertainty: number | null;
  event: "loss" | "checkpoint";
};

export const metricHeader =
  "time,training_percent,device,loss,loss_uncertainty,accuracy_percent,accuracy_uncertainty_percent,event,training_step\n";

function finite(value: string | undefined) {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function lossMetric(
  raw: string,
  percent: number,
  mlxTable = false,
): TrainingMetric | null {
  const loss = /\bloss=([\d.eE+-]+)(?:[±\uFFFD]([\d.eE+-]+))?/i.exec(raw);
  const mlx = mlxTable ? /^\s*\d[\d,]*\s+([\d.eE+-]+)\s/.exec(raw) : null;
  const value = finite(loss?.[1] ?? mlx?.[1]);
  if (value === null || percent < 0 || percent > 100) return null;
  // Older native builds print loss=0 before they have seen a supervised label.
  if (loss && value === 0 && /\bacc=-?nan\b/i.test(raw)) return null;
  const accuracy = /\bacc=([\d.eE+-]+)(?:[±\uFFFD]([\d.eE+-]+))?%?/i.exec(raw);
  const stepText =
    /\bdata\s*=\s*([\d,]+)\s*\//i.exec(raw)?.[1] ||
    (mlxTable ? /^\s*(\d[\d,]*)\s+/.exec(raw)?.[1] : undefined) ||
    /\b(?:iter|iteration)\s*[=:]?\s*([\d,]+)/i.exec(raw)?.[1];
  const step = stepText ? Number(stepText.replaceAll(",", "")) : null;
  return {
    time: new Date().toISOString(),
    percent,
    step:
      step !== null && Number.isSafeInteger(step) && step >= 0 ? step : null,
    device: /^\[([^\]]+)\]/.exec(raw)?.[1] || "Trainer",
    loss: value,
    lossUncertainty: finite(loss?.[2]),
    accuracy: finite(accuracy?.[1]),
    accuracyUncertainty: finite(accuracy?.[2]),
    event: "loss",
  };
}

export function metricRow(metric: TrainingMetric) {
  return (
    [
      metric.time,
      metric.percent.toFixed(4),
      metric.device,
      metric.loss ?? "",
      metric.lossUncertainty ?? "",
      metric.accuracy ?? "",
      metric.accuracyUncertainty ?? "",
      metric.event,
      metric.step ?? "",
    ].join(",") + "\n"
  );
}

export function parseMetricRow(raw: string): TrainingMetric | null {
  const fields = raw.split(",");
  if (
    ![8, 9].includes(fields.length) ||
    !["loss", "checkpoint"].includes(fields[7])
  )
    return null;
  const percent = Number(fields[1]);
  if (!Number.isFinite(percent) || percent < 0 || percent > 100) return null;
  const step = fields.length === 9 && fields[8] ? Number(fields[8]) : null;
  if (step !== null && (!Number.isSafeInteger(step) || step < 0)) return null;
  if (
    fields[7] === "loss" &&
    fields[3] === "0" &&
    fields[4] === "" &&
    fields[5] === "" &&
    fields[6] === ""
  )
    return null;
  return {
    time: fields[0],
    percent,
    step,
    device: fields[2],
    loss: finite(fields[3]),
    lossUncertainty: finite(fields[4]),
    accuracy: finite(fields[5]),
    accuracyUncertainty: finite(fields[6]),
    event: fields[7] as "loss" | "checkpoint",
  };
}
