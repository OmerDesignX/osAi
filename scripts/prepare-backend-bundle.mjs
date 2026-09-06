import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";

import { runtimeExecutable } from "./prepare-python-runtime.mjs";

const root = path.resolve(import.meta.dirname, "..");
const configurationFile = path.join(
  root,
  "releaseScripts",
  "backend-bundle.json",
);
const runtimeRoot = path.join(root, "build", "python-runtime");
const bundleRoot = path.join(root, "build", "backend-bundle");
const cacheRoot = path.join(root, "build", "backend-source-cache");
const nativeCacheRoot = path.join(root, "build", "backend-native-cache");
const workRoot = path.join(root, "build", "backend-bundle-work");
const MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024;
const NATIVE_CACHE_SCHEMA = 3;

export function trustedBackendArchiveUrl(raw) {
  try {
    const url = new URL(raw);
    return (
      url.protocol === "https:" &&
      (url.hostname === "codeload.github.com" ||
        url.hostname === "objects.githubusercontent.com")
    );
  } catch {
    return false;
  }
}

export function bundleTarget(platform, architecture) {
  if (!["macos", "windows", "linux"].includes(platform))
    throw new Error(`Unsupported backend bundle platform: ${platform}`);
  if (!["arm64", "x64"].includes(architecture))
    throw new Error(`Unsupported backend bundle architecture: ${architecture}`);
  if (platform !== "macos" && architecture !== "x64")
    throw new Error(`${platform} backend bundles currently support x64 only`);
  return `${platform}-${architecture}`;
}

function runtimePlatform(platform) {
  return platform;
}

function nativePlatform(platform) {
  if (platform === "macos") return "darwin";
  if (platform === "windows") return "win32";
  return "linux";
}

function executableSuffix(platform) {
  return platform === "windows" ? ".exe" : "";
}

function requirementsName(platform, architecture) {
  return platform === "macos" && architecture === "arm64"
    ? "requirements.txt"
    : platform === "linux"
      ? "requirements.txt"
      : "requirements-llama.txt";
}

async function sha256(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

async function directorySha256(directory) {
  const hash = createHash("sha256");
  async function visit(current, relative = "") {
    const entries = await fs.readdir(current, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      const childRelative = path.join(relative, entry.name);
      const child = path.join(current, entry.name);
      hash.update(childRelative);
      if (entry.isDirectory()) await visit(child, childRelative);
      else if (entry.isSymbolicLink()) hash.update(await fs.readlink(child));
      else
        for await (const chunk of createReadStream(child)) hash.update(chunk);
    }
  }
  await visit(directory);
  return hash.digest("hex");
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    if (!options.quiet)
      process.stdout.write(`\n> ${command} ${args.join(" ")}\n`);
    const child = spawn(command, args, {
      cwd: options.cwd || root,
      env: { ...process.env, ...(options.env || {}) },
      shell: false,
      windowsHide: true,
      stdio: options.capture ? ["ignore", "pipe", "pipe"] : "inherit",
    });
    let stdout = "";
    let stderr = "";
    if (options.capture) {
      child.stdout.on("data", (chunk) => {
        stdout += chunk;
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk;
      });
    }
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) resolve({ stdout: stdout.trim(), stderr: stderr.trim() });
      else
        reject(
          new Error(
            `${command} exited with code ${code}${stderr ? `: ${stderr.trim()}` : ""}`,
          ),
        );
    });
  });
}

export async function makePortableMacBinaries(directory) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await makePortableMacBinaries(file);
      continue;
    }
    if (entry.isSymbolicLink()) continue;
    if (!entry.name.endsWith(".dylib") && !entry.name.startsWith("llama-"))
      continue;
    const details = await run("otool", ["-l", file], {
      capture: true,
      quiet: true,
    });
    const rpaths = [];
    const lines = details.stdout.split("\n");
    for (let index = 0; index < lines.length; index += 1) {
      if (!lines[index].includes("LC_RPATH")) continue;
      const match = /\bpath\s+(.+?)\s+\(offset/.exec(lines[index + 2] || "");
      if (match) rpaths.push(match[1]);
    }
    for (const rpath of rpaths.filter((value) => value !== "@loader_path"))
      await run("install_name_tool", ["-delete_rpath", rpath, file], {
        quiet: true,
      });
    if (!rpaths.includes("@loader_path"))
      await run("install_name_tool", ["-add_rpath", "@loader_path", file], {
        quiet: true,
      });
  }
}

async function downloadArchive(url, destination) {
  if (!trustedBackendArchiveUrl(url))
    throw new Error("The osAi CLI source URL is not trusted");
  const response = await fetch(url, {
    redirect: "follow",
    headers: {
      Accept: "application/zip",
      "User-Agent": "osAi-App-release-builder",
    },
  });
  if (!response.ok || !response.body)
    throw new Error(`Could not download osAi CLI source (${response.status})`);
  if (!trustedBackendArchiveUrl(response.url))
    throw new Error("The osAi CLI source redirected to an untrusted host");
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > MAX_ARCHIVE_BYTES)
    throw new Error("The osAi CLI source archive is unexpectedly large");

  const temporary = `${destination}.${process.pid}.tmp`;
  await fs.rm(temporary, { force: true });
  const output = await fs.open(temporary, "wx", 0o600);
  const reader = response.body.getReader();
  let downloaded = 0;
  let failure;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      downloaded += value.byteLength;
      if (downloaded > MAX_ARCHIVE_BYTES)
        throw new Error("The osAi CLI source archive exceeded its size limit");
      let offset = 0;
      while (offset < value.byteLength) {
        const { bytesWritten } = await output.write(
          value,
          offset,
          value.byteLength - offset,
        );
        if (bytesWritten < 1)
          throw new Error("Could not write the osAi CLI source archive");
        offset += bytesWritten;
      }
      if (downloaded % (16 * 1024 * 1024) < value.byteLength)
        process.stdout.write(
          `Downloaded ${(downloaded / 1024 / 1024).toFixed(0)} MiB of osAi CLI source\n`,
        );
    }
  } catch (error) {
    failure = error;
  } finally {
    await reader.cancel().catch(() => undefined);
    await output.close();
  }
  if (failure) {
    await fs.rm(temporary, { force: true });
    throw failure;
  }
  if (downloaded < 1) throw new Error("The osAi CLI source archive is empty");
  await fs.rename(temporary, destination);
}

async function sourceRoot(extractionRoot) {
  const candidates = [
    extractionRoot,
    ...(await fs.readdir(extractionRoot, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(extractionRoot, entry.name)),
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
  throw new Error("The osAi CLI archive is incomplete");
}

function includeLocalSource(source, candidate) {
  const relative = path.relative(source, candidate);
  if (!relative) return true;
  const parts = relative.split(path.sep);
  if (
    parts.some((part) =>
      [".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__"].includes(
        part,
      ),
    )
  )
    return false;
  if (parts.includes("release-assets") || parts.includes("dist")) return false;
  if (parts[0] === "vendor" && parts[1] === "llama.cpp" && parts[2] === "build")
    return false;
  return !candidate.endsWith(".pyc") && !candidate.endsWith(".DS_Store");
}

function selectedAccelerator(platform) {
  const explicit = process.env.OSAI_RELEASE_LLAMA_ACCELERATOR?.toLowerCase();
  const allowed = new Set(["cpu", "metal", "cuda", "vulkan"]);
  if (explicit && !allowed.has(explicit))
    throw new Error(
      "OSAI_RELEASE_LLAMA_ACCELERATOR must be cpu, metal, cuda, or vulkan",
    );
  const accelerator = explicit || (platform === "macos" ? "metal" : "cpu");
  if (accelerator === "metal" && platform !== "macos")
    throw new Error("Metal release binaries can only be built on macOS");
  return accelerator;
}

export function releaseBuildJobs(totalMemory = os.totalmem()) {
  const explicit = Number(process.env.OSAI_RELEASE_BUILD_JOBS || "");
  if (Number.isInteger(explicit) && explicit > 0) return explicit;
  return Math.max(
    1,
    Math.min(4, Math.floor(totalMemory / (4 * 1024 * 1024 * 1024))),
  );
}

async function validateLlamaTargets(directory, platform, targets) {
  const suffix = executableSuffix(platform);
  for (const target of targets) {
    const candidates = [
      path.join(directory, `${target}${suffix}`),
      path.join(directory, "Release", `${target}${suffix}`),
    ];
    const found = (
      await Promise.all(
        candidates.map((file) => fs.stat(file).catch(() => null)),
      )
    ).find((details) => details?.isFile());
    if (!found) return false;
  }
  return true;
}

async function compileLlama(
  source,
  platform,
  architecture,
  sourceIdentity,
  configuration,
) {
  const host = nativePlatform(platform);
  if (process.platform !== host)
    throw new Error(`Build the ${platform} backend bundle on ${platform}`);
  const cmake = process.env.OSAI_CMAKE || "cmake";
  const llamaSource = path.join(source, "vendor", "llama.cpp");
  const build = path.join(workRoot, "llama-build");
  const accelerator = selectedAccelerator(platform);
  const jobs = releaseBuildJobs();
  const target = bundleTarget(platform, architecture);
  const cache = path.join(nativeCacheRoot, target);
  const cacheManifest = await fs
    .readFile(path.join(cache, "manifest.json"), "utf8")
    .then(JSON.parse)
    .catch(() => null);
  const destination = path.join(llamaSource, "build", "bin");
  if (
    cacheManifest?.schemaVersion === NATIVE_CACHE_SCHEMA &&
    cacheManifest?.sourceIdentity === sourceIdentity &&
    cacheManifest?.accelerator === accelerator &&
    (await validateLlamaTargets(
      path.join(cache, "bin"),
      platform,
      configuration.requiredLlamaTargets,
    ))
  ) {
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.cp(path.join(cache, "bin"), destination, {
      recursive: true,
      verbatimSymlinks: true,
    });
    if (platform === "macos") {
      await makePortableMacBinaries(destination);
      await fs.rm(path.join(cache, "bin"), { recursive: true, force: true });
      await fs.cp(destination, path.join(cache, "bin"), {
        recursive: true,
        verbatimSymlinks: true,
      });
    }
    process.stdout.write(
      `Using verified cached llama.cpp binaries for ${target}\n`,
    );
    return accelerator;
  }
  await fs.rm(build, { recursive: true, force: true });
  const configure = [
    "-S",
    llamaSource,
    "-B",
    build,
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
    "-DLLAMA_BUILD_APP=OFF",
    "-DLLAMA_CURL=OFF",
    "-DLLAMA_OPENSSL=OFF",
    "-DLLAMA_BUILD_EXAMPLES=ON",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BLAS=OFF",
    `-DGGML_METAL=${accelerator === "metal" ? "ON" : "OFF"}`,
    `-DGGML_CUDA=${accelerator === "cuda" ? "ON" : "OFF"}`,
    `-DGGML_VULKAN=${accelerator === "vulkan" ? "ON" : "OFF"}`,
  ];
  if (platform === "macos")
    configure.push(
      `-DCMAKE_OSX_ARCHITECTURES=${architecture === "x64" ? "x86_64" : "arm64"}`,
      "-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0",
      "-DCMAKE_BUILD_RPATH=@loader_path",
      "-DCMAKE_INSTALL_RPATH=@loader_path",
      "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
    );
  try {
    await run(cmake, configure, { cwd: llamaSource });
  } catch (error) {
    if (error?.code === "ENOENT")
      throw new Error(
        "CMake is required only on the release-building computer; it is never required by osAi App users",
      );
    throw error;
  }
  await run(
    cmake,
    [
      "--build",
      build,
      "--config",
      "Release",
      "--target",
      ...configuration.requiredLlamaTargets,
      "--parallel",
      String(jobs),
    ],
    { cwd: llamaSource },
  );

  await fs.rm(path.join(llamaSource, "build"), {
    recursive: true,
    force: true,
  });
  await fs.mkdir(path.dirname(destination), { recursive: true });
  await fs.cp(path.join(build, "bin"), destination, {
    recursive: true,
    verbatimSymlinks: true,
  });
  if (platform === "macos") await makePortableMacBinaries(destination);

  if (
    !(await validateLlamaTargets(
      destination,
      platform,
      configuration.requiredLlamaTargets,
    ))
  )
    throw new Error("One or more required llama.cpp targets were not built");
  await fs.rm(cache, { recursive: true, force: true });
  await fs.mkdir(cache, { recursive: true });
  await fs.cp(destination, path.join(cache, "bin"), {
    recursive: true,
    verbatimSymlinks: true,
  });
  await fs.writeFile(
    path.join(cache, "manifest.json"),
    `${JSON.stringify(
      {
        schemaVersion: NATIVE_CACHE_SCHEMA,
        sourceIdentity,
        accelerator,
        target,
      },
      null,
      2,
    )}\n`,
  );
  return accelerator;
}

async function prepareWheelhouse(source, platform, architecture) {
  const python = runtimeExecutable(runtimeRoot, runtimePlatform(platform));
  if (!(await fs.stat(python).catch(() => null))?.isFile())
    throw new Error(
      "Prepare the bundled Python runtime before the backend bundle",
    );
  const wheelhouse = path.join(bundleRoot, "wheelhouse");
  await fs.mkdir(wheelhouse, { recursive: true });
  const downloads =
    platform === "macos" && architecture === "arm64"
      ? [
          {
            requirements: "requirements-llama.txt",
            target: "macosx_12_0_arm64",
          },
          {
            requirements: "requirements.txt",
            target: "macosx_14_0_arm64",
          },
        ]
      : [
          {
            requirements: requirementsName(platform, architecture),
            target: platform === "macos" ? "macosx_12_0_x86_64" : undefined,
          },
        ];
  for (const download of downloads) {
    const requirements = path.join(
      source,
      "requirements",
      download.requirements,
    );
    if (!(await fs.stat(requirements).catch(() => null))?.isFile())
      throw new Error(`Backend requirements are missing: ${requirements}`);
    await run(python, [
      "-m",
      "pip",
      "download",
      "--disable-pip-version-check",
      "--only-binary=:all:",
      "--dest",
      wheelhouse,
      ...(download.target
        ? [
            "--platform",
            download.target,
            "--python-version",
            "312",
            "--implementation",
            "cp",
            "--abi",
            "cp312",
          ]
        : []),
      "--requirement",
      requirements,
      "pip",
      "setuptools",
      "wheel",
    ]);
  }
  const wheels = (await fs.readdir(wheelhouse)).filter((name) =>
    name.endsWith(".whl"),
  );
  if (wheels.length < 6)
    throw new Error("The offline Python wheelhouse is incomplete");
  return wheels.length;
}

export async function prepareBackendBundle(
  platform,
  architecture,
  { refreshSource = false } = {},
) {
  const target = bundleTarget(platform, architecture);
  const configuration = JSON.parse(
    await fs.readFile(configurationFile, "utf8"),
  );
  if (!trustedBackendArchiveUrl(configuration.archive))
    throw new Error(
      "releaseScripts/backend-bundle.json has an untrusted archive URL",
    );

  await fs.rm(bundleRoot, { recursive: true, force: true });
  await fs.rm(workRoot, { recursive: true, force: true });
  const source = path.join(bundleRoot, "source");
  await fs.mkdir(bundleRoot, { recursive: true });
  const localSourceValue = process.env.OSAI_CLI_SOURCE?.trim();
  let sourceIdentity;
  if (localSourceValue) {
    const localSource = await sourceRoot(path.resolve(localSourceValue));
    process.stdout.write(
      `Using local osAi CLI release source: ${localSource}\n`,
    );
    await fs.cp(localSource, source, {
      recursive: true,
      verbatimSymlinks: true,
      filter: (candidate) => includeLocalSource(localSource, candidate),
    });
    sourceIdentity = await directorySha256(source);
  } else {
    const archive = path.join(cacheRoot, "osai-cli-main.zip");
    await fs.mkdir(cacheRoot, { recursive: true });
    if (refreshSource) await fs.rm(archive, { force: true });
    if (!(await fs.stat(archive).catch(() => null))?.isFile()) {
      process.stdout.write(
        "Downloading the backend source used by this release\n",
      );
      await downloadArchive(configuration.archive, archive);
    } else {
      process.stdout.write("Using the cached osAi CLI source archive\n");
    }
    sourceIdentity = await sha256(archive);
    const extraction = path.join(workRoot, "source-archive");
    await fs.mkdir(extraction, { recursive: true });
    const python = runtimeExecutable(runtimeRoot, runtimePlatform(platform));
    await run(python, ["-m", "zipfile", "-e", archive, extraction]);
    const extractedSource = await sourceRoot(extraction);
    await fs.cp(extractedSource, source, {
      recursive: true,
      verbatimSymlinks: true,
    });
  }

  const accelerator = await compileLlama(
    source,
    platform,
    architecture,
    sourceIdentity,
    configuration,
  );
  const wheelCount = await prepareWheelhouse(source, platform, architecture);
  await fs.writeFile(
    path.join(bundleRoot, "OSAI_BACKEND_BUNDLE.json"),
    `${JSON.stringify(
      {
        schemaVersion: 1,
        target,
        repository: configuration.repository,
        ref: configuration.ref,
        sourceIdentity,
        llamaAccelerator: accelerator,
        requiredLlamaTargets: configuration.requiredLlamaTargets,
        wheelCount,
      },
      null,
      2,
    )}\n`,
  );
  await fs.rm(workRoot, { recursive: true, force: true });
  process.stdout.write(
    `Prepared self-contained osAi backend bundle for ${target} (${wheelCount} Python wheels, llama.cpp ${accelerator})\n`,
  );
}

export async function cleanPreparedBackendBundle() {
  await fs.rm(bundleRoot, { recursive: true, force: true });
  await fs.rm(workRoot, { recursive: true, force: true });
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(import.meta.filename)
) {
  const [platform, architecture, option] = process.argv.slice(2);
  if (platform === "clean") {
    await cleanPreparedBackendBundle();
    process.stdout.write("Removed the generated backend bundle\n");
    process.exit(0);
  }
  if (!platform || !architecture)
    throw new Error(
      "Usage: node scripts/prepare-backend-bundle.mjs <macos|windows|linux> <arm64|x64> [--refresh-source]",
    );
  await prepareBackendBundle(platform, architecture, {
    refreshSource: option === "--refresh-source",
  });
}
