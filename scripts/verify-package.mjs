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
      if (code === 0) resolve(`${stdout}${stderr ? `\n${stderr}` : ""}`.trim());
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

async function verifyBackendResources(
  resources,
  platform,
  architecture,
  expectedNativeArchitecture,
) {
  const backend = path.join(resources, "backend");
  const bundle = JSON.parse(
    await fs.readFile(path.join(backend, "OSAI_BACKEND_BUNDLE.json"), "utf8"),
  );
  if (bundle.target !== `${platform}-${architecture}`)
    throw new Error(
      `Backend bundle targets ${bundle.target}; expected ${platform}-${architecture}`,
    );
  await requireArtifact(
    path.join(backend, "source", "scripts", "setup_osai.py"),
    1_000,
  );
  const wheelhouse = path.join(backend, "wheelhouse");
  const wheels = (await fs.readdir(wheelhouse)).filter((name) =>
    name.endsWith(".whl"),
  );
  if (wheels.length < 6)
    throw new Error("The packaged offline Python wheelhouse is incomplete");
  const suffix = platform === "windows" ? ".exe" : "";
  let llamaCompletion = "";
  for (const target of bundle.requiredLlamaTargets || []) {
    const direct = path.join(
      backend,
      "source",
      "vendor",
      "llama.cpp",
      "build",
      "bin",
      `${target}${suffix}`,
    );
    const release = path.join(
      path.dirname(direct),
      "Release",
      path.basename(direct),
    );
    const executable = (await fs.stat(direct).catch(() => null))?.isFile()
      ? direct
      : release;
    // Recent llama.cpp builds keep most implementation code in adjacent shared
    // libraries, so a valid launcher can be only a few tens of kilobytes.
    await requireArtifact(executable, 10_000);
    if (target === "llama-completion") llamaCompletion = executable;
    if (platform === "macos" && expectedNativeArchitecture) {
      const detected = await run("lipo", ["-archs", executable]);
      if (detected !== expectedNativeArchitecture)
        throw new Error(
          `${target} contains architecture ${detected}; expected ${expectedNativeArchitecture}`,
        );
    }
  }
  if (platform === "macos") {
    const bin = path.join(
      backend,
      "source",
      "vendor",
      "llama.cpp",
      "build",
      "bin",
    );
    const libraries = (await fs.readdir(bin)).filter((name) =>
      name.endsWith(".dylib"),
    );
    for (const library of libraries) {
      const linked = await run("otool", ["-L", path.join(bin, library)]);
      for (const line of linked.split("\n").slice(1)) {
        const dependency = line.trim().split(/\s+/, 1)[0] || "";
        if (
          dependency &&
          !dependency.startsWith("@") &&
          !dependency.startsWith("/usr/lib/") &&
          !dependency.startsWith("/System/Library/")
        )
          throw new Error(
            `${library} depends on unpackaged library ${dependency}`,
          );
      }
    }
    if (!llamaCompletion)
      throw new Error("The packaged llama-completion executable is missing");
    const version = await run("/usr/bin/env", [
      "-i",
      "PATH=/usr/bin:/bin",
      llamaCompletion,
      "--version",
    ]);
    if (!version.includes("version:"))
      throw new Error("The packaged llama.cpp executable did not start");
    const loadCommands = await run("otool", ["-l", llamaCompletion]);
    const minimumMatch = loadCommands.match(/\bminos\s+([0-9.]+)/);
    if (!minimumMatch || Number(minimumMatch[1].split(".")[0]) > 12)
      throw new Error(
        `The packaged llama.cpp executable requires unsupported macOS ${minimumMatch?.[1] || "unknown"}`,
      );
  }
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
    const resources = path.join(application, "Contents", "Resources");

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
    await verifyBackendResources(resources, "macos", architecture, expected);

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
  await verifyBackendResources(
    path.join(packageDirectory, "win-unpacked", "resources"),
    "windows",
    architecture,
  );
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
  await verifyBackendResources(
    path.join(packageDirectory, "linux-unpacked", "resources"),
    "linux",
    architecture,
  );
  console.log("Verified " + packages[0]);
} else {
  throw new Error(
    "Usage: node scripts/verify-package.mjs <macos|windows|linux> <architecture> [package-directory]",
  );
}
