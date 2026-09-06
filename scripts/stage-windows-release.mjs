import {
  copyFile,
  lstat,
  mkdir,
  readdir,
  readFile,
  rm,
  stat,
} from "node:fs/promises";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const manifest = JSON.parse(
  await readFile(path.join(root, "package.json"), "utf8"),
);
const releaseRoot = path.resolve(
  process.env.OSAI_PACKAGE_DIR || path.join(root, "release"),
);
const outputRoot = path.resolve(root, "release-assets", "windows");
const expectedOutputRoot = path.join(root, "release-assets", "windows");

if (outputRoot !== expectedOutputRoot)
  throw new Error("Refusing to stage outside " + expectedOutputRoot);

const artifactName = "osAi-Setup-" + manifest.version + ".exe";
const source = path.join(releaseRoot, artifactName);
const sourceStats = await stat(source);

if (!sourceStats.isFile() || sourceStats.size < 50_000_000)
  throw new Error(artifactName + " is missing or unexpectedly small");

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

const destination = path.join(outputRoot, artifactName);
await copyFile(source, destination);
const destinationStats = await stat(destination);
if (destinationStats.size !== sourceStats.size)
  throw new Error(artifactName + " was not copied completely");

console.log("Staged " + artifactName + " in " + outputRoot);
