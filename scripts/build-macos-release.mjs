import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");

if (process.platform !== "darwin")
  throw new Error("The macOS release must be built on macOS");

function run(command, args, env = {}) {
  return new Promise((resolve, reject) => {
    process.stdout.write("\n> " + command + " " + args.join(" ") + "\n");
    const child = spawn(command, args, {
      cwd: root,
      stdio: "inherit",
      shell: false,
      env: { ...process.env, ...env },
    });
    child.once("error", reject);
    child.once("exit", (code) =>
      code === 0
        ? resolve()
        : reject(new Error(command + " exited with code " + code)),
    );
  });
}

async function makeDirectoriesWritable(directory) {
  const details = await fs.lstat(directory).catch(() => null);
  if (!details) return;
  if (!details.isDirectory() || details.isSymbolicLink())
    throw new Error("Refusing to clean unexpected release path: " + directory);
  await fs.chmod(directory, 0o700);
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    if (entry.isDirectory() && !entry.isSymbolicLink())
      await makeDirectoriesWritable(path.join(directory, entry.name));
  }
}

async function removeGeneratedRelease(target) {
  const releaseRoot = path.join(root, "release");
  const relative = path.relative(releaseRoot, target);
  if (relative.startsWith("..") || path.isAbsolute(relative))
    throw new Error("Refusing to clean outside " + releaseRoot);
  if (!(await fs.lstat(target).catch(() => null))) return;
  await makeDirectoriesWritable(target);
  await fs.rm(target, { recursive: true, force: true });
}

await run("pnpm", ["run", "release:check-disk"]);
await run("bash", ["releaseScripts/macos/prepare-icon.sh"]);
await run("pnpm", ["run", "format:check"]);
await run("pnpm", ["test"]);
await run("pnpm", ["exec", "vite", "build"], {
  NODE_OPTIONS: "--max-old-space-size=4096",
});

for (const architecture of ["arm64", "x64"]) {
  await run(process.execPath, [
    "scripts/prepare-python-runtime.mjs",
    "macos",
    architecture,
  ]);
  await run(process.execPath, [
    "scripts/prepare-backend-bundle.mjs",
    "macos",
    architecture,
    ...(architecture === "arm64" ? ["--refresh-source"] : []),
  ]);
  const packageDirectory = path.join(root, "release", "macos-" + architecture);
  await removeGeneratedRelease(packageDirectory);

  await run(
    "pnpm",
    [
      "exec",
      "electron-builder",
      "--mac",
      "dmg",
      "--" + architecture,
      "--config.directories.output=" + packageDirectory,
      "--publish",
      "never",
    ],
    { CSC_IDENTITY_AUTO_DISCOVERY: "false" },
  );

  await run(process.execPath, [
    "scripts/verify-package.mjs",
    "macos",
    architecture,
    packageDirectory,
  ]);
}

await run("pnpm", ["run", "release:stage:macos"]);
await run(process.execPath, ["scripts/prepare-python-runtime.mjs", "clean"]);
await run(process.execPath, ["scripts/prepare-backend-bundle.mjs", "clean"]);
await removeGeneratedRelease(path.join(root, "release"));

process.stdout.write(
  "\nApple Silicon and Intel DMGs verified in release-assets/macos; intermediate release/ removed.\n",
);
