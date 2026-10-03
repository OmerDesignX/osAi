import type { TrainingRequest } from "./types.js";

// Keep the renderer's ready state in step with the main process approval check.
// Dataset contents and device inventory are verified separately before training.
export function calibrationKey(input: TrainingRequest) {
  return JSON.stringify([
    input.modelSource,
    input.modelVersion,
    input.tier,
    input.customModelFolder,
    input.engine,
    input.accelerator,
    input.multiGpu,
    input.devices,
    input.fineTuneData,
    input.fullContentContext,
    input.optimizer,
    input.iterations,
    input.scale,
    input.dropout,
    input.seed,
    input.gradientAccumulationSteps,
    input.gradientCheckpoint,
    input.maskPrompt,
    input.splitMode,
    input.tensorSplit,
    input.mainGpu,
    input.distributedWorkers,
  ]);
}
