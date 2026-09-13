import type {
  HardwareInfo,
  LoraTargetModule,
  TrainingRequest,
} from "./types.js";

const GIB = 1024 ** 3;
const compactTargets: LoraTargetModule[] = ["mlp.down_proj"];
const balancedTargets: LoraTargetModule[] = [
  "self_attn.q_proj",
  "self_attn.v_proj",
  "mlp.down_proj",
];
const allTargets: LoraTargetModule[] = [
  "self_attn.q_proj",
  "self_attn.k_proj",
  "self_attn.v_proj",
  "self_attn.o_proj",
  "mlp.gate_proj",
  "mlp.up_proj",
  "mlp.down_proj",
];

const modelBytes = {
  "llama.cpp": {
    small: 2_708_804_288,
    medium: 3_464_055_456,
    large: 4_482_403_136,
  },
  mlx: {
    small: 2_912_931_406,
    medium: 3_701_329_697,
    large: 4_489_728_089,
  },
} as const;

export type HardwarePreset = {
  profile: "compact" | "balanced" | "performance" | "maximum";
  batchSize: number;
  maxSeqLength: number;
  numLayers: number;
  rank: number;
  ggufBatchSize: number;
  ggufThreads: number;
  targetModules: LoraTargetModule[];
};

export function selectHardwarePreset(
  hardware: HardwareInfo,
  engine: TrainingRequest["engine"],
  tier: TrainingRequest["tier"],
): HardwarePreset {
  // Match the CLI's cross-platform automatic engine choice. An Apple-silicon
  // build previews MLX; other automatic builds preview llama.cpp.
  const runtime =
    engine === "mlx" ||
    (engine === "auto" &&
      hardware.platform === "darwin" &&
      hardware.architecture === "arm64")
      ? "mlx"
      : "llama.cpp";
  const total = Math.max(hardware.physicalMemoryBytes, 1);
  const budget = total * 0.75;
  const size = modelBytes[runtime][tier];
  const headroom = budget - size;
  const profile =
    total <= 10 * GIB || headroom < 4 * GIB
      ? "compact"
      : total <= 20 * GIB || headroom < 8 * GIB
        ? "balanced"
        : total <= 40 * GIB || headroom < 16 * GIB
          ? "performance"
          : "maximum";

  const common = {
    compact: [1, 64, 1, 2, 8, 2, compactTargets],
    balanced: [1, 128, 2, 4, 8, 4, balancedTargets],
    performance: [2, 256, 4, 8, 16, 8, allTargets],
    maximum: [4, 1024, 8, 16, 32, 16, allTargets],
  } as const;
  const selected = common[profile];
  let batchSize: number = selected[0];
  let maxSeqLength: number = selected[1];
  let numLayers: number = selected[2];
  let rank: number = selected[3];
  let ggufBatchSize: number = selected[4];
  const threads: number = selected[5];
  let targets: readonly LoraTargetModule[] = selected[6];

  if (runtime === "llama.cpp") {
    const gguf = {
      compact: [256, 1, 1, 8, compactTargets],
      balanced: [256, 1, 2, 8, compactTargets],
      performance: [256, 2, 4, 16, balancedTargets],
      maximum: [256, 4, 8, 16, balancedTargets],
    } as const;
    [maxSeqLength, numLayers, rank, ggufBatchSize, targets] = gguf[profile];
  }

  return {
    profile,
    batchSize,
    maxSeqLength,
    numLayers,
    rank,
    ggufBatchSize,
    ggufThreads: Math.min(threads, Math.max(1, hardware.logicalCpuCount)),
    targetModules: [...targets],
  };
}
