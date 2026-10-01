import type { TrainingMetric } from "./types.js";

export type LossAxis = "progress" | "time" | "steps";

const colors = ["#89cff0", "#e8b879", "#a8d59b", "#c9a8ed"];

function elapsedLabel(milliseconds: number) {
  const seconds = Math.round(milliseconds / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = String(seconds % 60).padStart(2, "0");
  return minutes < 60
    ? `${minutes}:${remainder}`
    : `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${remainder}`;
}

export function LossChart({
  metrics,
  axis,
}: {
  metrics: TrainingMetric[];
  axis: LossAxis;
}) {
  const lossPoints = metrics.filter(
    (point): point is TrainingMetric & { loss: number } =>
      point.event === "loss" &&
      point.loss !== null &&
      Number.isFinite(point.loss),
  );
  const firstTimestamp = Math.min(
    ...lossPoints
      .map((point) => Date.parse(point.time))
      .filter(Number.isFinite),
  );
  const valueFor = (point: TrainingMetric): number | null => {
    if (axis === "progress") return point.percent;
    if (axis === "steps")
      return point.step !== null &&
        point.step !== undefined &&
        Number.isFinite(point.step)
        ? point.step
        : null;
    const timestamp = Date.parse(point.time);
    return Number.isFinite(firstTimestamp) && Number.isFinite(timestamp)
      ? Math.max(0, timestamp - firstTimestamp)
      : null;
  };
  const samples = lossPoints
    .map((point) => ({ point, value: valueFor(point) }))
    .filter(
      (
        sample,
      ): sample is { point: (typeof lossPoints)[number]; value: number } =>
        sample.value !== null,
    );
  if (!samples.length) {
    const message =
      axis === "steps" && lossPoints.length
        ? "This session has no recorded trainer steps. New runs record them."
        : axis === "time" && lossPoints.length
          ? "This session has no recorded timestamps for a time plot."
          : "Loss will appear after the first training update.";
    return <p className="loss-chart-empty">{message}</p>;
  }

  const devices = [...new Set(samples.map(({ point }) => point.device))];
  const losses = samples.map(({ point }) => point.loss);
  const lowest = Math.min(...losses);
  const highest = Math.max(...losses);
  const spread = Math.max(0.05, highest - lowest);
  const minLoss = Math.max(0, lowest - spread * 0.12);
  const maxLoss = highest + spread * 0.12;
  const checkpointEntries = metrics
    .filter((point) => point.event === "checkpoint")
    .map((point) => ({ point, value: valueFor(point) }))
    .filter(
      (entry): entry is { point: TrainingMetric; value: number } =>
        entry.value !== null,
    );
  const largestValue = Math.max(
    ...samples.map((sample) => sample.value),
    ...checkpointEntries.map((entry) => entry.value),
  );
  const maxValue =
    axis === "progress"
      ? Math.min(100, Math.max(1, Math.ceil(largestValue * 1.05)))
      : axis === "time"
        ? Math.max(60_000, largestValue * 1.05)
        : Math.max(1, Math.ceil(largestValue * 1.05));
  const left = 54;
  const top = 16;
  const width = 478;
  const height = 206;
  const x = (value: number) =>
    left + (Math.min(value, maxValue) / maxValue) * width;
  const y = (loss: number) =>
    top + (1 - (loss - minLoss) / (maxLoss - minLoss)) * height;
  const checkpoints = checkpointEntries.filter(
    (entry) => entry.value <= maxValue,
  );
  const tickLabel = (value: number) =>
    axis === "progress"
      ? `${value.toFixed(maxValue < 10 ? 1 : 0)}%`
      : axis === "time"
        ? elapsedLabel(value)
        : Math.round(value).toLocaleString();
  const xTitle =
    axis === "progress"
      ? "Training progress"
      : axis === "time"
        ? "Time since first loss update"
        : "Trainer data / iteration step";

  return (
    <div className="loss-chart-content">
      <svg
        viewBox="0 0 550 290"
        role="img"
        aria-label={`Training loss against ${xTitle.toLowerCase()}, with checkpoint markers`}
      >
        {[0, 0.5, 1].map((fraction) => (
          <g key={fraction}>
            <line
              className="loss-chart-grid"
              x1={left}
              x2={left + width}
              y1={top + fraction * height}
              y2={top + fraction * height}
            />
            <text
              className="loss-chart-axis"
              x={left - 7}
              y={top + fraction * height + 4}
              textAnchor="end"
            >
              {(maxLoss - fraction * (maxLoss - minLoss)).toFixed(2)}
            </text>
          </g>
        ))}
        {[0, 0.5, 1].map((fraction) => (
          <text
            key={fraction}
            className="loss-chart-axis"
            x={left + fraction * width}
            y={top + height + 20}
            textAnchor={
              fraction === 0 ? "start" : fraction === 1 ? "end" : "middle"
            }
          >
            {tickLabel(fraction * maxValue)}
          </text>
        ))}
        <text
          className="loss-chart-axis loss-chart-axis-title"
          x={left + width / 2}
          y={top + height + 43}
          textAnchor="middle"
        >
          {xTitle}
        </text>
        <text
          className="loss-chart-axis loss-chart-axis-title"
          x="12"
          y={top + height / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${top + height / 2})`}
        >
          Loss
        </text>
        {checkpoints.map(({ point, value }, index) => (
          <line
            key={`${point.time}-${index}`}
            className="loss-chart-checkpoint"
            x1={x(value)}
            x2={x(value)}
            y1={top}
            y2={top + height}
          >
            <title>Checkpoint saved at {tickLabel(value)}</title>
          </line>
        ))}
        {devices.map((device, index) => {
          const points = samples.filter(({ point }) => point.device === device);
          const color = colors[index % colors.length];
          return (
            <g key={device}>
              <polyline
                points={points
                  .map(({ point, value }) => `${x(value)},${y(point.loss)}`)
                  .join(" ")}
                fill="none"
                stroke={color}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {points.length === 1 && (
                <circle
                  cx={x(points[0].value)}
                  cy={y(points[0].point.loss)}
                  r="3"
                  fill={color}
                />
              )}
            </g>
          );
        })}
      </svg>
      <div className="loss-chart-legend" aria-label="Chart legend">
        {devices.map((device, index) => (
          <span key={device}>
            <i style={{ backgroundColor: colors[index % colors.length] }} />
            {device} loss
          </span>
        ))}
        {checkpoints.length > 0 && (
          <span>
            <i className="checkpoint-key" />
            Saved checkpoint
          </span>
        )}
      </div>
    </div>
  );
}
