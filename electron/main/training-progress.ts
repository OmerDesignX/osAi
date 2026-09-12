export type TrainingStage = "fine-tuning" | "alignment" | "fine-tune-align";

export interface FineTuneProgress {
  completed: number;
  total: number;
}

export function cleanTerminalLine(raw: string) {
  return raw
    .replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g, "")
    .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "")
    .replace(/\u0008/g, "")
    .trim();
}

function count(value: string | undefined) {
  if (!value) return 0;
  const parsed = Number(value.replaceAll(",", ""));
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : 0;
}

function validProgress(completed: number, total: number) {
  return total > 0 && completed >= 0 && completed <= total;
}

export function phaseProgress(
  completed: number,
  total: number,
  phase: "fine-tuning" | "alignment",
  stage: TrainingStage,
) {
  const ratio = Math.max(0, Math.min(1, completed / Math.max(1, total)));
  if (stage === "fine-tune-align")
    return phase === "fine-tuning" ? 5 + ratio * 58 : 72 + ratio * 25;
  return 5 + ratio * 92;
}

export class FineTuneProgressParser {
  private plannedSteps = 0;
  private mlxTable = false;

  constructor(private readonly fallbackTotal = 1) {}

  consume(raw: string): FineTuneProgress | null {
    const line = cleanTerminalLine(raw);
    if (!line) return null;

    const plan = /\bosai:\s*training plan\b.*?\bsteps\s*=\s*([\d,]+)/i.exec(
      line,
    );
    if (plan) {
      this.plannedSteps = count(plan[1]);
      return this.plannedSteps
        ? { completed: 0, total: this.plannedSteps }
        : null;
    }

    if (/^iter\s+train_loss\s+tok\/s\s+tokens$/i.test(line)) {
      this.mlxTable = true;
      return null;
    }

    const iteration =
      /\b(?:iter|iteration|epoch)\s*[=:]?\s*([\d,]+)(?:\s*\/\s*([\d,]+))?/i.exec(
        line,
      );
    const trainProgress = line.toLowerCase().includes("train")
      ? /\b([\d,]+)\s*\/\s*([\d,]+)\b/.exec(line)
      : null;
    const mlxRow = this.mlxTable
      ? /^(\d[\d,]*)\s+(?:nan|inf|-?(?:\d+(?:\.\d+)?|\.\d+))\b/i.exec(line)
      : null;
    const match = iteration || trainProgress || mlxRow;
    if (!match) return null;

    const completed = count(match[1]);
    const total =
      count(match[2]) || this.plannedSteps || Math.max(1, this.fallbackTotal);
    return validProgress(completed, total) ? { completed, total } : null;
  }
}

export function recoverFineTuneProgress(output: string, fallbackTotal = 1) {
  const parser = new FineTuneProgressParser(fallbackTotal);
  let latest: FineTuneProgress | null = null;
  for (const line of output.split(/\r\n|\r|\n/)) {
    const progress = parser.consume(line);
    if (progress && progress.completed >= (latest?.completed ?? -1))
      latest = progress;
  }
  return latest;
}
