import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");
const manifestFile = path.join(root, "releaseScripts", "python-runtime.json");
const runtimeRoot = path.join(root, "build", "python-runtime");
const cacheRoot = path.join(root, "build", "python-runtime-cache");
const MAX_ARCHIVE_BYTES = 100 * 1024 * 1024;

export function trustedPythonRuntimeUrl(raw) {
  try {
    const url = new URL(raw);
    return (
      url.protocol === "https:" &&
      (url.hostname === "github.com" ||
        url.hostname === "objects.githubusercontent.com" ||
        url.hostname === "release-assets.githubusercontent.com")
    );
  } catch {
    return false;
  }
}

export function runtimeExecutable(directory, platform) {
  return path.join(
    directory,
    platform === "windows" ? "python.exe" : path.join("bin", "python3"),
  );
}

async function sha256(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: root,
      shell: false,
      windowsHide: true,
      stdio: "inherit",
    });
    child.once("error", reject);
    child.once("close", (code) =>
      code === 0
        ? resolve()
        : reject(new Error(`${command} exited with code ${code}`)),
    );
  });
}

async function download(url, destination) {
  const response = await fetch(url, {
    redirect: "follow",
    headers: { "User-Agent": "osAi-App-release-builder/0.1.0" },
  });
  if (!response.ok || !response.body)
    throw new Error(`Could not download bundled Python (${response.status})`);
  if (!trustedPythonRuntimeUrl(response.url))
    throw new Error(
      "The bundled Python download redirected to an untrusted host",
    );
  const declaredSize = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredSize) && declaredSize > MAX_ARCHIVE_BYTES)
    throw new Error("The bundled Python archive is unexpectedly large");

  const temporary = `${destination}.${process.pid}.tmp`;
  const handle = await fs.open(temporary, "wx", 0o600);
  const reader = response.body.getReader();
  let downloaded = 0;
  let failure;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      downloaded += value.byteLength;
      if (downloaded > MAX_ARCHIVE_BYTES)
        throw new Error("The bundled Python archive exceeded its size limit");
      let offset = 0;
      while (offset < value.byteLength) {
        const { bytesWritten } = await handle.write(
          value,
          offset,
          value.byteLength - offset,
        );
        if (bytesWritten < 1)
          throw new Error("Could not write the bundled Python archive");
        offset += bytesWritten;
      }
      if (downloaded % (8 * 1024 * 1024) < value.byteLength)
        process.stdout.write(
          `Downloaded ${(downloaded / 1024 / 1024).toFixed(0)} MiB\n`,
        );
    }
  } catch (error) {
    failure = error;
  } finally {
    await reader.cancel().catch(() => undefined);
    await handle.close();
  }
  if (failure) {
    await fs.rm(temporary, { force: true });
    throw failure;
  }
  if (downloaded < 1) throw new Error("The bundled Python download was empty");
  await fs.rename(temporary, destination);
}

export async function prepareRuntime(platform, architecture) {
  const manifest = JSON.parse(await fs.readFile(manifestFile, "utf8"));
  const key = `${platform}-${architecture}`;
  const selected = manifest.runtimes[key];
  if (!selected)
    throw new Error(`No bundled Python runtime is defined for ${key}`);
  const url = `https://github.com/astral-sh/python-build-standalone/releases/download/${manifest.release}/${encodeURIComponent(selected.file)}`;
  const archive = path.join(cacheRoot, selected.file);
  await fs.mkdir(cacheRoot, { recursive: true });
  const cachedHash = await sha256(archive).catch(() => "");
  if (cachedHash !== selected.sha256) {
    if (cachedHash) await fs.rm(archive, { force: true });
    process.stdout.write(
      `Downloading CPython ${manifest.pythonVersion} for ${key}\n`,
    );
    await download(url, archive);
  } else {
    process.stdout.write(
      `Using verified cached CPython ${manifest.pythonVersion} for ${key}\n`,
    );
  }
  const archiveHash = await sha256(archive);
  if (archiveHash !== selected.sha256)
    throw new Error(`Bundled Python SHA-256 mismatch for ${key}`);

  await fs.rm(runtimeRoot, { recursive: true, force: true });
  await fs.mkdir(runtimeRoot, { recursive: true, mode: 0o700 });
  const tar =
    platform === "windows" && process.platform === "win32"
      ? path.join(
          process.env.SystemRoot || "C:\\Windows",
          "System32",
          "tar.exe",
        )
      : "tar";
  await run(tar, ["-xzf", archive, "-C", runtimeRoot, "--strip-components=1"]);
  const executable = runtimeExecutable(runtimeRoot, platform);
  const details = await fs.stat(executable).catch(() => null);
  if (!details?.isFile())
    throw new Error(`Bundled Python executable is missing for ${key}`);
  if (platform !== "windows") await fs.chmod(executable, 0o755);
  await run(executable, [
    "-c",
    `import sys; assert sys.version_info[:3] == (${manifest.pythonVersion.replaceAll(".", ", ")}); print(sys.version.split()[0])`,
  ]);
  await fs.writeFile(
    path.join(runtimeRoot, "OSAI_RUNTIME.json"),
    `${JSON.stringify(
      {
        provider: manifest.provider,
        license: manifest.license,
        pythonVersion: manifest.pythonVersion,
        release: manifest.release,
        target: key,
        archive: selected.file,
        sha256: selected.sha256,
      },
      null,
      2,
    )}\n`,
  );
  process.stdout.write(
    `Prepared verified CPython ${manifest.pythonVersion} for ${key}\n`,
  );
}

export async function cleanPreparedRuntime() {
  await fs.rm(runtimeRoot, { recursive: true, force: true });
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(import.meta.filename)
) {
  const [platform, architecture] = process.argv.slice(2);
  if (platform === "clean") {
    await cleanPreparedRuntime();
    process.stdout.write("Removed the generated Python runtime directory\n");
    process.exit(0);
  }
  if (!platform || !architecture)
    throw new Error(
      "Usage: node scripts/prepare-python-runtime.mjs <macos|windows|linux> <arm64|x64>",
    );
  await prepareRuntime(platform, architecture);
}
