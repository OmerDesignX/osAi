import type { SessionState } from "./types.js";

type ProgressState = Pick<
  SessionState,
  "phase" | "trainingPercent" | "progress" | "indeterminate"
>;

export function displayedSessionProgress(state: ProgressState) {
  if (state.indeterminate) return { percent: null, label: "Working" };
  if (state.phase === "fine-tuning" || state.phase === "alignment") {
    const percent = state.trainingPercent;
    if (percent === undefined || !Number.isFinite(percent))
      return { percent: null, label: "Working" };
    const exact = Math.max(0, Math.min(100, percent));
    return { percent: exact, label: formatTrainingPercent(exact) };
  }
  const progress = Math.max(0, Math.min(100, state.progress));
  return { percent: progress, label: `${progress.toFixed(1)}%` };
}

export function formatTrainingPercent(percent: number) {
  const exact = Math.max(0, Math.min(100, percent));
  return exact > 0 && exact < 0.01 ? "<0.01%" : `${exact.toFixed(2)}%`;
}
