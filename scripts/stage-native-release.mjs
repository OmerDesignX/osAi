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

if (process.argv[2] !== "macos")
  throw new Error("Usage: node scripts/stage-native-release.mjs macos");

const root = path.resolve(import.meta.dirname, "..");
const manifest = JSON.parse(
  await readFile(path.join(root, "package.json"), "utf8"),
);
const outputRoot = path.resolve(root, "release-assets", "macos");
const expectedOutputRoot = path.join(root, "release-assets", "macos");

if (outputRoot !== expectedOutputRoot)
  throw new Error("Refusing to stage outside " + expectedOutputRoot);

const artifacts = ["arm64", "x64"].map((architecture) => {
  const artifactName =
    "osAi-" + manifest.version + "-mac-" + architecture + ".dmg";
  return {
    artifactName,
    source: path.join(root, "release", "macos-" + architecture, artifactName),
  };
});

const sourceStats = new Map();
for (const artifact of artifacts) {
  const details = await stat(artifact.source);
  if (!details.isFile() || details.size < 50_000_000)
    throw new Error(
      artifact.artifactName + " is missing or unexpectedly small",
    );
  sourceStats.set(artifact.artifactName, details);
}

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

for (const artifact of artifacts) {
  const destination = path.join(outputRoot, artifact.artifactName);
  await copyFile(artifact.source, destination);
  const destinationStats = await stat(destination);
  if (destinationStats.size !== sourceStats.get(artifact.artifactName).size)
    throw new Error(artifact.artifactName + " was not copied completely");
  console.log("Staged " + artifact.artifactName + " in " + outputRoot);
}
