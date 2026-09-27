import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { runPnpm } from "../releaseScripts/common/run-pnpm.mjs";

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

await runPnpm(["run", "release:check-disk"], { cwd: root });
await run("bash", ["releaseScripts/macos/prepare-icon.sh"]);
await runPnpm(["run", "format:check"], { cwd: root });
await runPnpm(["test"], { cwd: root });
await runPnpm(["exec", "vite", "build"], {
  cwd: root,
  env: { NODE_OPTIONS: "--max-old-space-size=4096" },
});

for (const architecture of ["arm64", "x64"]) {
  await run(process.execPath, [
    "scripts/prepare-python-runtime.mjs",
    "macos",
    architecture,
  ]);
  await run(process.execPath, [
    "scripts/prepare-native-tools.mjs",
    "macos",
    architecture,
  ]);
  const packageDirectory = path.join(root, "release", "macos-" + architecture);
  await removeGeneratedRelease(packageDirectory);

  await runPnpm(
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
    { cwd: root, env: { CSC_IDENTITY_AUTO_DISCOVERY: "false" } },
  );

  await run(process.execPath, [
    "scripts/verify-package.mjs",
    "macos",
    architecture,
    packageDirectory,
  ]);
}

await runPnpm(["run", "release:stage:macos"], { cwd: root });
await run(process.execPath, ["scripts/prepare-python-runtime.mjs", "clean"]);
await run(process.execPath, ["scripts/prepare-native-tools.mjs", "clean"]);
await removeGeneratedRelease(path.join(root, "release"));

process.stdout.write(
  "\nApple Silicon and Intel DMGs verified in release-assets/macos; intermediate release/ removed.\n",
);
