import { copyFile, lstat, mkdir, readdir, rm, stat } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const releaseRoot = path.resolve(
  process.env.OSAI_PACKAGE_DIR || path.join(root, "release"),
);
const outputRoot = path.resolve(root, "release-assets", "linux");
const entries = await readdir(releaseRoot, { withFileTypes: true });
const packages = entries.filter(
  (entry) => entry.isFile() && /^osAi-.+-x64\.deb$/i.test(entry.name),
);

if (packages.length !== 1)
  throw new Error("Expected exactly one versioned x64 Linux .deb package");

const source = path.join(releaseRoot, packages[0].name);
const sourceStats = await stat(source);
if (sourceStats.size < 20_000_000)
  throw new Error(packages[0].name + " is unexpectedly small");

await mkdir(outputRoot, { recursive: true });
for (const entry of await readdir(outputRoot, { withFileTypes: true })) {
  const target = path.resolve(outputRoot, entry.name);
  if (path.dirname(target) !== outputRoot)
    throw new Error("Refusing to clean unexpected release path: " + target);
  const targetStats = await lstat(target);
  if (targetStats.isSymbolicLink())
    throw new Error("Refusing to remove linked release output: " + target);
  await rm(target, { recursive: targetStats.isDirectory(), force: true });
}

const destination = path.join(outputRoot, packages[0].name);
await copyFile(source, destination);
if ((await stat(destination)).size !== sourceStats.size)
  throw new Error(packages[0].name + " was not copied completely");

console.log("Staged " + packages[0].name + " in " + outputRoot);
