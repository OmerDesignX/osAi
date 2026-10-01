import type { TrainingMetric } from "./training-metrics.js";

type Reading = { at: number; percent: number; loss: number };

const minimumTimeMs = 5 * 60_000;
const windowSize = 12;
const windowCount = 4;

/** Stops only after four sustained, rising averages for every reporting device. */
export class LossRiseDetector {
  private enabled = false;
  private firstReadingAt = 0;
  private readonly readings = new Map<string, Reading[]>();

  setEnabled(enabled: boolean) {
    if (this.enabled === enabled) return;
    this.enabled = enabled;
    this.firstReadingAt = 0;
    this.readings.clear();
  }

  ready(at = Date.now()) {
    return (
      this.enabled &&
      this.firstReadingAt > 0 &&
      at - this.firstReadingAt >= minimumTimeMs &&
      this.readings.size > 0 &&
      [...this.readings.values()].every((series) => {
        const recent = series.slice(-windowSize * windowCount);
        return (
          recent.length === windowSize * windowCount &&
          recent.at(-1)!.percent - recent[0].percent >= 1
        );
      })
    );
  }

  observe(metric: TrainingMetric, at = Date.now()): string | null {
    if (!this.enabled || metric.event !== "loss" || metric.loss === null)
      return null;
    if (!Number.isFinite(metric.loss) || metric.loss <= 0) return null;
    this.firstReadingAt ||= at;
    const readings = this.readings.get(metric.device) || [];
    const previous = readings.at(-1);
    // Native trainers may redraw the same progress line many times.
    if (
      previous &&
      metric.percent <= previous.percent &&
      at - previous.at < 10_000
    )
      return null;
    readings.push({ at, percent: metric.percent, loss: metric.loss });
    if (readings.length > 256) readings.shift();
    this.readings.set(metric.device, readings);
    if (!this.ready(at) || metric.percent < 12) return null;

    const trends = [...this.readings.values()].map((series) => {
      const recent = series.slice(-windowSize * windowCount);
      if (
        recent.length < windowSize * windowCount ||
        recent.at(-1)!.percent - recent[0].percent < 1
      )
        return null;
      const averages = Array.from({ length: windowCount }, (_, index) => {
        const window = recent.slice(
          index * windowSize,
          (index + 1) * windowSize,
        );
        return window.reduce((sum, point) => sum + point.loss, 0) / windowSize;
      });
      if (
        !averages
          .slice(1)
          .every((value, index) => value > averages[index] * 1.006) ||
        averages[3] < averages[0] * 1.025
      )
        return null;
      return averages;
    });
    if (trends.some((trend) => !trend)) return null;
    const first =
      trends.reduce((sum, trend) => sum + trend![0], 0) / trends.length;
    const last =
      trends.reduce((sum, trend) => sum + trend![3], 0) / trends.length;
    return `average loss rose from ${first.toFixed(4)} to ${last.toFixed(4)} across four windows`;
  }
}
