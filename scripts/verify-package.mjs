import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const [platform, architecture, packageDirectoryValue] = process.argv.slice(2);
const root = path.resolve(import.meta.dirname, "..");
const packageDirectory = path.resolve(
  packageDirectoryValue || path.join(root, "release"),
);
const manifest = JSON.parse(
  await fs.readFile(path.join(root, "package.json"), "utf8"),
);

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: root,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.once("error", reject);
    child.once("exit", (code) => {
      if (code === 0) resolve(stdout.trim());
      else
        reject(
          new Error(
            command +
              " exited with code " +
              code +
              (stderr ? ": " + stderr.trim() : ""),
          ),
        );
    });
  });
}

async function requireArtifact(file, minimumBytes) {
  const details = await fs.stat(file);
  if (!details.isFile() || details.size < minimumBytes)
    throw new Error(path.basename(file) + " is missing or unexpectedly small");
  return details;
}

if (platform === "macos") {
  if (process.platform !== "darwin")
    throw new Error("macOS packages must be verified on macOS");
  if (!["arm64", "x64"].includes(architecture))
    throw new Error("Expected macOS architecture arm64 or x64");

  const artifactName =
    "osAi-" + manifest.version + "-mac-" + architecture + ".dmg";
  const artifact = path.join(packageDirectory, artifactName);
  await requireArtifact(artifact, 50_000_000);

  const mountPoint = await fs.mkdtemp(path.join(os.tmpdir(), "osai-dmg-"));
  let mounted = false;
  try {
    await run("hdiutil", [
      "attach",
      "-nobrowse",
      "-readonly",
      "-mountpoint",
      mountPoint,
      artifact,
    ]);
    mounted = true;

    const application = path.join(mountPoint, "osAi.app");
    const executable = path.join(application, "Contents", "MacOS", "osAi");
    const archive = path.join(application, "Contents", "Resources", "app.asar");
    const python = path.join(
      application,
      "Contents",
      "Resources",
      "python",
      "bin",
      "python3",
    );
    // Electron's macOS launcher is intentionally small; the application code
    // lives in app.asar. Validate both independently instead of applying the
    // Windows/Linux executable-size threshold to the launcher.
    await requireArtifact(executable, 10_000);
    await requireArtifact(archive, 1_000_000);
    await requireArtifact(python, 1_000_000);

    const detected = await run("lipo", ["-archs", executable]);
    const expected = architecture === "x64" ? "x86_64" : "arm64";
    if (detected.split(/\s+/).length !== 1 || detected !== expected)
      throw new Error(
        artifactName +
          " contains architecture " +
          detected +
          "; expected " +
          expected,
      );

    const pythonArchitecture = await run("lipo", ["-archs", python]);
    if (
      pythonArchitecture.split(/\s+/).length !== 1 ||
      pythonArchitecture !== expected
    )
      throw new Error(
        artifactName +
          " bundles Python for " +
          pythonArchitecture +
          "; expected " +
          expected,
      );
    const pythonVersion = await run(python, ["--version"]);
    if (!pythonVersion.startsWith("Python 3.12."))
      throw new Error(artifactName + " bundles unexpected " + pythonVersion);

    const minimum = await run("plutil", [
      "-extract",
      "LSMinimumSystemVersion",
      "raw",
      path.join(application, "Contents", "Info.plist"),
    ]);
    if (minimum !== "12.0")
      throw new Error(
        artifactName + " requires unexpected macOS version " + minimum,
      );
  } finally {
    if (mounted) await run("hdiutil", ["detach", mountPoint]);
    await fs.rm(mountPoint, { recursive: true, force: true });
  }

  console.log("Verified " + artifactName + " (" + architecture + ")");
} else if (platform === "windows") {
  const artifactName = "osAi-Setup-" + manifest.version + ".exe";
  await requireArtifact(path.join(packageDirectory, artifactName), 50_000_000);
  const python = path.join(
    packageDirectory,
    "win-unpacked",
    "resources",
    "python",
    "python.exe",
  );
  await requireArtifact(python, 1_000_000);
  const pythonVersion = await run(python, ["--version"]);
  if (!pythonVersion.startsWith("Python 3.12."))
    throw new Error(artifactName + " bundles unexpected " + pythonVersion);
  console.log("Verified " + artifactName);
} else if (platform === "linux") {
  const entries = await fs.readdir(packageDirectory);
  const packages = entries.filter((entry) => /^osAi-.+-x64\.deb$/i.test(entry));
  if (packages.length !== 1)
    throw new Error("Expected exactly one versioned x64 Linux package");
  await requireArtifact(path.join(packageDirectory, packages[0]), 20_000_000);
  const python = path.join(
    packageDirectory,
    "linux-unpacked",
    "resources",
    "python",
    "bin",
    "python3",
  );
  await requireArtifact(python, 1_000_000);
  const pythonVersion = await run(python, ["--version"]);
  if (!pythonVersion.startsWith("Python 3.12."))
    throw new Error(packages[0] + " bundles unexpected " + pythonVersion);
  console.log("Verified " + packages[0]);
} else {
  throw new Error(
    "Usage: node scripts/verify-package.mjs <macos|windows|linux> <architecture> [package-directory]",
  );
}
