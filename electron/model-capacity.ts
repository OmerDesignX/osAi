const GIB = 1024 ** 3;

export function modelFitsShardedGpuMemory(
  sizeBytes: number,
  freeBytes: number[],
): boolean | null {
  const known = freeBytes.filter(
    (value) => Number.isFinite(value) && value > 0,
  );
  if (known.length === 0) return null;
  const usable = known.reduce(
    (total, value) =>
      total + Math.max(0, value - Math.max(512 * 1024 ** 2, value / 10)),
    0,
  );
  // Match the CLI compact profile. The reserve is shared across the complete
  // sharded model, not charged once per GPU, so several small cards remain useful.
  return usable >= sizeBytes + 2 * GIB;
}
