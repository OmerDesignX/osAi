import fs from "node:fs/promises";
import path from "node:path";

export async function findSourceRoot(extractedRoot: string) {
  const candidates = [
    extractedRoot,
    ...(await fs
      .readdir(extractedRoot, { withFileTypes: true })
      .then((entries) =>
        entries
          .filter((entry) => entry.isDirectory())
          .map((entry) => path.join(extractedRoot, entry.name)),
      )),
  ];
  for (const candidate of candidates) {
    const required = [
      path.join(candidate, "pyproject.toml"),
      path.join(candidate, "scripts", "setup_osai.py"),
      path.join(candidate, "vendor", "llama.cpp", "CMakeLists.txt"),
    ];
    if (
      (
        await Promise.all(
          required.map((file) => fs.stat(file).catch(() => null)),
        )
      ).every((details) => details?.isFile())
    )
      return candidate;
  }
  throw new Error(
    "The downloaded archive is not a complete osAi CLI repository",
  );
}

export async function managedBackendSource(executable: string) {
  const extractedRoot = path.resolve(
    path.dirname(executable),
    "..",
    "..",
    "source",
  );
  return findSourceRoot(extractedRoot).catch(() => null);
}

export async function backendRuntimeEnvironment(
  executable: string,
  base: NodeJS.ProcessEnv = process.env,
) {
  const source = await managedBackendSource(executable);
  return source ? { ...base, OSAI_ROOT: source } : { ...base };
}
