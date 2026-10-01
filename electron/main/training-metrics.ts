export type TrainingMetric = {
  time: string;
  percent: number;
  device: string;
  loss: number | null;
  lossUncertainty: number | null;
  accuracy: number | null;
  accuracyUncertainty: number | null;
  event: "loss" | "checkpoint";
};

export const metricHeader =
  "time,training_percent,device,loss,loss_uncertainty,accuracy_percent,accuracy_uncertainty_percent,event\n";

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
  const accuracy = /\bacc=([\d.eE+-]+)(?:[±\uFFFD]([\d.eE+-]+))?%?/i.exec(raw);
  return {
    time: new Date().toISOString(),
    percent,
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
    ].join(",") + "\n"
  );
}

export function parseMetricRow(raw: string): TrainingMetric | null {
  const fields = raw.split(",");
  if (fields.length !== 8 || !["loss", "checkpoint"].includes(fields[7]))
    return null;
  const percent = Number(fields[1]);
  if (!Number.isFinite(percent) || percent < 0 || percent > 100) return null;
  return {
    time: fields[0],
    percent,
    device: fields[2],
    loss: finite(fields[3]),
    lossUncertainty: finite(fields[4]),
    accuracy: finite(fields[5]),
    accuracyUncertainty: finite(fields[6]),
    event: fields[7] as "loss" | "checkpoint",
  };
}
