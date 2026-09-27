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
  if (!source) return { ...base };
  const environment: NodeJS.ProcessEnv = { ...base, OSAI_ROOT: source };
  const oldPath =
    Object.entries(environment).find(
      ([key]) => key.toLowerCase() === "path",
    )?.[1] || "";
  for (const key of Object.keys(environment))
    if (key.toLowerCase() === "path") delete environment[key];
  const resources = process.resourcesPath;
  const suffix = process.platform === "win32" ? ".exe" : "";
  const bundledTools = resources
    ? [
        path.join(resources, "native-tools", "cmake", "bin"),
        path.join(resources, "native-tools", "bin"),
      ]
    : [];
  environment.PATH = [path.dirname(executable), ...bundledTools, oldPath]
    .filter(Boolean)
    .join(path.delimiter);
  environment.OSAI_CMAKE = resources
    ? path.join(resources, "native-tools", "cmake", "bin", `cmake${suffix}`)
    : path.join(path.dirname(executable), `cmake${suffix}`);
  return environment;
}
