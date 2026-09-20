import fs from "node:fs/promises";
import path from "node:path";

type LegacyVariant = {
  runtime: "mlx" | "llama.cpp";
  tier: "small" | "medium" | "large";
  folder: string;
  required: string[];
};

const legacyVariants: LegacyVariant[] = [
  ...(
    [
      ["small", "osCode-MLX-Small-Q5", 21],
      ["medium", "osCode-MLX-Medium-Q6", 27],
      ["large", "osCode-MLX-Large-Q8", 34],
    ] as const
  ).map(([tier, folder, shards]) => ({
    runtime: "mlx" as const,
    tier,
    folder: path.join("MLX", folder),
    required: [
      "config.json",
      "model.safetensors.index.json",
      "tokenizer.json",
      "tokenizer_config.json",
      ...Array.from(
        { length: shards },
        (_, index) =>
          `model-${String(index + 1).padStart(5, "0")}-of-${String(shards).padStart(5, "0")}.safetensors`,
      ),
    ],
  })),
  ...(
    [
      ["small", "osCode-GGUF-Small-Q4_K_M", 2],
      ["medium", "osCode-GGUF-Medium-Q6_K", 2],
      ["large", "osCode-GGUF-Large-Q8_0", 3],
    ] as const
  ).map(([tier, name, shards]) => ({
    runtime: "llama.cpp" as const,
    tier,
    folder: path.join("GGUF", tier),
    required: Array.from(
      { length: shards },
      (_, index) =>
        `${name}-${String(index + 1).padStart(5, "0")}-of-${String(shards).padStart(5, "0")}.gguf`,
    ),
  })),
];

async function isCompleteV1Model(source: string, variant: LegacyVariant) {
  const directory = await fs.lstat(source).catch(() => null);
  if (!directory?.isDirectory() || directory.isSymbolicLink()) return false;
  let manifest: unknown;
  try {
    manifest = JSON.parse(
      await fs.readFile(path.join(source, "OSCODE_MODEL.json"), "utf8"),
    );
  } catch {
    return false;
  }
  if (!manifest || typeof manifest !== "object") return false;
  const data = manifest as Record<string, unknown>;
  const primary = variant.required.find((file) => file.endsWith(".gguf"));
  const repositoryPath =
    variant.runtime === "mlx"
      ? variant.folder.split(path.sep).join("/")
      : `GGUF/${primary}`;
  if (
    ![1, 2].includes(data.schema_version as number) ||
    (data.model_version !== undefined && data.model_version !== "v1") ||
    data.runtime !== variant.runtime ||
    data.tier !== variant.tier ||
    data.repository_path !== repositoryPath ||
    !Array.isArray(data.files)
  )
    return false;

  const seen = new Set<string>();
  for (const record of data.files) {
    if (!record || typeof record !== "object") return false;
    const file = record as Record<string, unknown>;
    if (
      typeof file.path !== "string" ||
      !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(file.path) ||
      seen.has(file.path) ||
      !Number.isSafeInteger(file.bytes) ||
      (file.bytes as number) < 1 ||
      typeof file.sha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(file.sha256)
    )
      return false;
    const details = await fs
      .lstat(path.join(source, file.path))
      .catch(() => null);
    if (
      !details?.isFile() ||
      details.isSymbolicLink() ||
      details.size !== file.bytes
    )
      return false;
    seen.add(file.path);
  }
  return variant.required.every((file) => seen.has(file));
}

/** Promote complete V1 downloads without changing paths saved in older sessions. */
export async function migrateLegacyV1Models(
  root: string,
  activeSession = false,
) {
  if (activeSession) return [];
  const location = path.resolve(root);
  const details = await fs.lstat(location).catch(() => null);
  if (!details?.isDirectory() || details.isSymbolicLink()) return [];
  const moved: string[] = [];
  for (const variant of legacyVariants) {
    const source = path.join(location, variant.folder);
    const destination = path.join(location, "V1", variant.folder);
    const sourceParent = await fs.lstat(path.dirname(source)).catch(() => null);
    if (!sourceParent?.isDirectory() || sourceParent.isSymbolicLink()) continue;
    const versionRoot = await fs
      .lstat(path.join(location, "V1"))
      .catch(() => null);
    if (
      versionRoot &&
      (!versionRoot.isDirectory() || versionRoot.isSymbolicLink())
    )
      continue;
    const destinationParent = await fs
      .lstat(path.dirname(destination))
      .catch(() => null);
    if (
      destinationParent &&
      (!destinationParent.isDirectory() || destinationParent.isSymbolicLink())
    )
      continue;
    if ((await fs.lstat(destination).catch(() => null)) !== null) continue;
    if (!(await isCompleteV1Model(source, variant))) continue;
    const files = await fs.readdir(source, { withFileTypes: true });
    if (files.some((file) => !file.isFile() || file.isSymbolicLink())) continue;
    await fs.mkdir(path.dirname(destination), { recursive: true, mode: 0o700 });
    const staging = await fs.mkdtemp(
      path.join(path.dirname(destination), ".v1-migration-"),
    );
    try {
      // Hard links share the bytes on the same volume. Old absolute paths
      // still work for saved sessions, without a second multi-GB download.
      for (const file of files)
        await fs.link(
          path.join(source, file.name),
          path.join(staging, file.name),
        );
      if ((await fs.lstat(destination).catch(() => null)) !== null) continue;
      await fs.rename(staging, destination);
    } finally {
      // Discard only our uniquely named staging folder after an interrupted
      // promotion; the legacy model remains untouched.
      await fs.rm(staging, { recursive: true, force: true });
    }
    moved.push(variant.folder);
  }
  return moved;
}
