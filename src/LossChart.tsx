import type { TrainingMetric } from "./types.js";

const colors = ["#89cff0", "#e8b879", "#a8d59b", "#c9a8ed"];

export function LossChart({ metrics }: { metrics: TrainingMetric[] }) {
  const samples = metrics.filter(
    (point): point is TrainingMetric & { loss: number } =>
      point.event === "loss" &&
      point.loss !== null &&
      Number.isFinite(point.loss),
  );
  if (!samples.length)
    return (
      <p className="loss-chart-empty">
        Loss will appear after the first training update.
      </p>
    );

  const devices = [...new Set(samples.map((point) => point.device))];
  const losses = samples.map((point) => point.loss);
  const lowest = Math.min(...losses);
  const highest = Math.max(...losses);
  const spread = Math.max(0.05, highest - lowest);
  const minLoss = Math.max(0, lowest - spread * 0.12);
  const maxLoss = highest + spread * 0.12;
  const maxPercent = Math.min(
    100,
    Math.max(
      1,
      Math.ceil(Math.max(...samples.map((point) => point.percent)) * 1.05),
    ),
  );
  const left = 46;
  const top = 16;
  const width = 486;
  const height = 218;
  const x = (percent: number) =>
    left + (Math.min(percent, maxPercent) / maxPercent) * width;
  const y = (loss: number) =>
    top + (1 - (loss - minLoss) / (maxLoss - minLoss)) * height;
  const checkpoints = metrics.filter(
    (point) => point.event === "checkpoint" && point.percent <= maxPercent,
  );

  return (
    <div className="loss-chart-content">
      <svg
        viewBox="0 0 550 270"
        role="img"
        aria-label="Training loss by progress, with checkpoint markers"
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
            {(fraction * maxPercent).toFixed(maxPercent < 10 ? 1 : 0)}%
          </text>
        ))}
        {checkpoints.map((point, index) => (
          <line
            key={`${point.time}-${index}`}
            className="loss-chart-checkpoint"
            x1={x(point.percent)}
            x2={x(point.percent)}
            y1={top}
            y2={top + height}
          >
            <title>Checkpoint saved at {point.percent.toFixed(2)}%</title>
          </line>
        ))}
        {devices.map((device, index) => {
          const points = samples.filter((point) => point.device === device);
          const color = colors[index % colors.length];
          return (
            <g key={device}>
              <polyline
                points={points
                  .map((point) => `${x(point.percent)},${y(point.loss)}`)
                  .join(" ")}
                fill="none"
                stroke={color}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {points.length === 1 && (
                <circle
                  cx={x(points[0].percent)}
                  cy={y(points[0].loss)}
                  r="3"
                  fill={color}
                />
              )}
            </g>
          );
        })}
      </svg>
      <div className="loss-chart-legend">
        {devices.map((device, index) => (
          <span key={device}>
            <i style={{ backgroundColor: colors[index % colors.length] }} />
            {device}
          </span>
        ))}
        {checkpoints.length > 0 && (
          <span>
            <i className="checkpoint-key" />
            Checkpoint
          </span>
        )}
      </div>
    </div>
  );
}
