import fs from "node:fs/promises";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..", "..");
const versionFile = path.join(root, "releaseScripts", "VERSION.txt");
const rootVersionFile = path.join(root, "VERSION.txt");
const packageFile = path.join(root, "package.json");
const version = (await fs.readFile(versionFile, "utf8")).trim();

if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version))
  throw new Error(
    "releaseScripts/VERSION.txt must contain one semantic version such as 0.1.1",
  );

const manifest = JSON.parse(await fs.readFile(packageFile, "utf8"));
if (manifest.name !== "osai-app")
  throw new Error("Refusing to update an unexpected package at " + packageFile);

let changed = false;
if (manifest.version !== version) {
  manifest.version = version;
  await fs.writeFile(packageFile, JSON.stringify(manifest, null, 2) + "\n");
  changed = true;
}
if (
  (await fs.readFile(rootVersionFile, "utf8").catch(() => "")).trim() !==
  version
) {
  await fs.writeFile(rootVersionFile, version + "\n");
  changed = true;
}

console.log(
  changed
    ? "Synchronized osAi version " + version
    : "Version " + version + " is already synchronized",
);
