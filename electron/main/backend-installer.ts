import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { BackendInstallStatus } from "../types.js";
import { findSourceRoot, managedBackendSource } from "./backend-source.js";
import { osAiVersionFromOutput } from "./session-service.js";

export { findSourceRoot } from "./backend-source.js";

const PYTHON_CHECK =
  "import sys; print('.'.join(map(str, sys.version_info[:3]))); " +
  "raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)";
const MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024;
const MAX_EXTRACTED_BYTES = 12 * 1024 * 1024 * 1024;
export const EXTRACT_ARCHIVE = `import pathlib, shutil, stat, sys, zipfile
archive, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).resolve()
destination.mkdir(parents=True, exist_ok=True)
total = 0
with zipfile.ZipFile(archive) as source:
    members = source.infolist()
    if len(members) > 150000:
        raise ValueError("CLI archive contains too many entries")
    for member in members:
        name = member.filename
        parts = pathlib.PurePosixPath(name).parts
        if not parts or name.startswith("/") or "\\\\" in name or any(part in ("", ".", "..") or ":" in part for part in parts):
            raise ValueError("CLI archive contains an unsafe path")
        if stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK:
            continue
        total += member.file_size
        if total > ${MAX_EXTRACTED_BYTES}:
            raise ValueError("CLI archive exceeds its extracted size limit")
        # osAi builds llama.cpp without its server. The server UI contains
        # paths beyond Windows MAX_PATH and is not used by the CLI trainer.
        if parts[1:5] == ("vendor", "llama.cpp", "tools", "ui"):
            continue
        target = destination.joinpath(*parts).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("CLI archive escaped the install directory")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
print("Verified and extracted osAi CLI source")`;

type PythonCommand = {
  executable: string;
  prefix: string[];
  version: string;
};

type CommandResult = {
  code: number;
  stdout: string;
  stderr: string;
};

export function backendExecutablePath(
  venv: string,
  platform = process.platform,
) {
  return path.join(
    venv,
    platform === "win32" ? "Scripts" : "bin",
    platform === "win32" ? "osai.exe" : "osai",
  );
}

export function bundledPythonExecutable(
  runtimeRoot: string,
  platform = process.platform,
) {
  return path.join(
    runtimeRoot,
    platform === "win32" ? "python.exe" : path.join("bin", "python3"),
  );
}

type BackendSourceManifest = {
  repository: string;
  archive: string;
  ref: string;
  revision: string;
};

export function trustedBackendArchiveUrl(raw: string) {
  try {
    const url = new URL(raw);
    return (
      url.protocol === "https:" &&
      url.username === "" &&
      url.password === "" &&
      url.port === "" &&
      url.hostname === "codeload.github.com" &&
      url.pathname === "/OmerDesignX/osAi-CLI/zip/refs/heads/main"
    );
  } catch {
    return false;
  }
}

async function readSourceManifest(file: string) {
  try {
    const value = JSON.parse(
      await fs.readFile(file, "utf8"),
    ) as BackendSourceManifest;
    if (
      value.repository !== "https://github.com/OmerDesignX/osAi-CLI" ||
      !trustedBackendArchiveUrl(value.archive) ||
      value.ref !== "main" ||
      !value.revision
    )
      return null;
    return value;
  } catch {
    return null;
  }
}

export async function backendInstallationIsCurrent(
  executable: string,
  installationsRoot: string,
  sourceManifestPath: string,
  platform = process.platform,
) {
  const root = path.resolve(installationsRoot);
  const resolvedExecutable = path.resolve(executable);
  const relative = path.relative(root, resolvedExecutable);
  if (
    !relative ||
    relative.startsWith(`..${path.sep}`) ||
    relative === ".." ||
    path.isAbsolute(relative)
  )
    return false;

  const [installId] = relative.split(path.sep);
  const installRoot = path.join(root, installId);
  const expectedExecutable = path.resolve(
    backendExecutablePath(path.join(installRoot, ".venv"), platform),
  );
  if (resolvedExecutable !== expectedExecutable) return false;

  const [expected, installed, source] = await Promise.all([
    readSourceManifest(sourceManifestPath),
    readSourceManifest(path.join(installRoot, "OSAI_BACKEND_SOURCE.json")),
    managedBackendSource(executable),
  ]);
  return Boolean(
    expected &&
    installed &&
    source &&
    expected.archive === installed.archive &&
    expected.ref === installed.ref &&
    expected.revision === installed.revision,
  );
}

function runCommand(
  executable: string,
  args: string[],
  options: {
    cwd?: string;
    env?: NodeJS.ProcessEnv;
    timeoutMs?: number;
    onLine?: (line: string) => void;
  } = {},
) {
  return new Promise<CommandResult>((resolve, reject) => {
    const child = spawn(executable, args, {
      cwd: options.cwd,
      env: options.env,
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let remainder = "";
    let settled = false;
    const timeout = options.timeoutMs
      ? setTimeout(() => {
          child.kill();
          reject(new Error(`Command timed out: ${executable}`));
        }, options.timeoutMs)
      : null;
    const finish = (callback: () => void) => {
      if (settled) return;
      settled = true;
      if (timeout) clearTimeout(timeout);
      callback();
    };
    const report = (text: string) => {
      if (!options.onLine) return;
      remainder += text;
      const lines = remainder.split(/\r?\n/);
      remainder = lines.pop() || "";
      for (const line of lines) if (line.trim()) options.onLine(line.trim());
    };
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      report(text);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      report(text);
    });
    child.once("error", (error) => finish(() => reject(error)));
    child.once("close", (code) =>
      finish(() => {
        if (remainder.trim()) options.onLine?.(remainder.trim());
        resolve({ code: code ?? -1, stdout, stderr });
      }),
    );
  });
}

async function verifyBundledPython(
  runtimeRoot: string,
): Promise<PythonCommand> {
  const executable = bundledPythonExecutable(runtimeRoot);
  const stat = await fs.stat(executable).catch(() => null);
  if (!stat?.isFile())
    throw new Error("This osAi App build does not contain its Python runtime");
  const result = await runCommand(executable, ["-c", PYTHON_CHECK], {
    timeoutMs: 15_000,
  });
  if (result.code !== 0)
    throw new Error("The bundled Python runtime is damaged or unsupported");
  return {
    executable,
    prefix: [],
    version: result.stdout.trim().split(/\s+/).at(-1) || "3.12",
  };
}

function cleanSetupLine(line: string) {
  return line.replace(/\s+/g, " ").trim().slice(0, 220);
}

export function backendSetupFailureMessage(
  log: string,
  code: number,
  logPath: string,
) {
  const details = log.trim().slice(-16_000);
  return [
    `osAi CLI setup failed (exit code ${code}).`,
    details,
    `Full setup log: ${logPath}`,
  ]
    .filter(Boolean)
    .join("\n\n");
}

async function downloadSource(
  archiveUrl: string,
  destination: string,
  progress: (percent?: number) => void,
) {
  const response = await fetch(archiveUrl, {
    redirect: "follow",
    signal: AbortSignal.timeout(30 * 60 * 1000),
  });
  if (!response.ok || !response.body || !trustedBackendArchiveUrl(response.url))
    throw new Error(
      `Could not download the trusted osAi CLI source (${response.status})`,
    );
  const expectedBytes = Number(response.headers.get("content-length") || 0);
  if (expectedBytes > MAX_ARCHIVE_BYTES)
    throw new Error("The osAi CLI source archive exceeds its download limit");
  const file = await fs.open(destination, "wx", 0o600);
  const hash = createHash("sha256");
  let bytes = 0;
  try {
    for await (const chunk of response.body) {
      bytes += chunk.byteLength;
      if (bytes > MAX_ARCHIVE_BYTES)
        throw new Error(
          "The osAi CLI source archive exceeded its download limit",
        );
      let offset = 0;
      while (offset < chunk.byteLength) {
        const { bytesWritten } = await file.write(
          chunk,
          offset,
          chunk.byteLength - offset,
        );
        if (bytesWritten < 1)
          throw new Error("Could not write the CLI archive");
        offset += bytesWritten;
      }
      hash.update(chunk);
      if (expectedBytes)
        progress(Math.min(60, Math.round((bytes / expectedBytes) * 60)));
    }
  } finally {
    await file.close();
  }
  if (!bytes) throw new Error("The osAi CLI source archive is empty");
  return hash.digest("hex");
}

export class BackendInstaller {
  private status: BackendInstallStatus = {
    state: "idle",
    message: "osAi CLI is not installed",
  };
  private active: Promise<BackendInstallStatus> | null = null;

  constructor(
    private readonly installationsRoot: string,
    private readonly pythonRuntimeRoot: string,
    private readonly sourceManifestPath: string,
    private readonly emit: (status: BackendInstallStatus) => void,
    private readonly selectExecutable: (executable: string) => Promise<void>,
  ) {}

  getStatus() {
    return { ...this.status };
  }

  isCurrent(executable: string) {
    return backendInstallationIsCurrent(
      executable,
      this.installationsRoot,
      this.sourceManifestPath,
    );
  }

  private update(next: BackendInstallStatus) {
    this.status = next;
    this.emit(this.getStatus());
  }

  install() {
    if (this.active) return this.active;
    this.active = this.performInstall().finally(() => {
      this.active = null;
    });
    return this.active;
  }

  private async performInstall() {
    const installId = new Date().toISOString().replace(/[-:.TZ]/g, "");
    const installRoot = path.join(this.installationsRoot, installId);
    const archive = path.join(installRoot, "osai-cli.zip");
    const extractedSource = path.join(installRoot, "source");
    const venv = path.join(installRoot, ".venv");
    let log = "";
    let lastStatusAt = 0;
    try {
      this.update({
        state: "preparing-python",
        message: "Checking the Python 3.12 runtime",
      });
      const python = await verifyBundledPython(this.pythonRuntimeRoot);
      const sourceManifest = await readSourceManifest(this.sourceManifestPath);
      if (!sourceManifest)
        throw new Error(
          "This osAi App build has no trusted CLI download configuration",
        );
      await fs.mkdir(installRoot, { recursive: true, mode: 0o700 });
      this.update({
        state: "downloading",
        message: "Downloading osAi CLI source",
      });
      const digest = await downloadSource(
        sourceManifest.archive,
        archive,
        (percent) => {
          this.update({
            state: "downloading",
            message: "Downloading osAi CLI source",
            percent,
          });
        },
      );
      this.update({
        state: "extracting",
        message: "Verifying and extracting osAi CLI source",
        percent: 65,
      });
      const extraction = await runCommand(
        python.executable,
        ["-c", EXTRACT_ARCHIVE, archive, extractedSource],
        { timeoutMs: 15 * 60 * 1000 },
      );
      log = (log + extraction.stdout + extraction.stderr).slice(-2_000_000);
      if (extraction.code !== 0)
        throw new Error(
          "Could not extract osAi CLI source: " +
            cleanSetupLine(
              extraction.stderr.trim().split(/\r?\n/).filter(Boolean).at(-1) ||
                "Archive verification failed",
            ),
        );
      await fs.rm(archive);
      const source = await findSourceRoot(extractedSource);

      const installLog = path.join(installRoot, "install.log");
      const inheritedPath = Object.entries(process.env).find(
        ([key]) => key.toLowerCase() === "path",
      )?.[1];
      const setupEnvironment: NodeJS.ProcessEnv = {
        ...process.env,
        OSAI_ROOT: source,
        OSAI_CMAKE: path.join(
          path.dirname(this.pythonRuntimeRoot),
          "native-tools",
          "cmake",
          "bin",
          process.platform === "win32" ? "cmake.exe" : "cmake",
        ),
        PATH: [
          path.join(
            path.dirname(this.pythonRuntimeRoot),
            "native-tools",
            "cmake",
            "bin",
          ),
          path.join(
            path.dirname(this.pythonRuntimeRoot),
            "native-tools",
            "bin",
          ),
          inheritedPath || "",
        ].join(path.delimiter),
        DO_NOT_TRACK: "1",
        HF_HUB_DISABLE_TELEMETRY: "1",
        PIP_DISABLE_PIP_VERSION_CHECK: "1",
        PIP_NO_INPUT: "1",
        TOKENIZERS_PARALLELISM: "false",
        WANDB_MODE: "disabled",
      };
      for (const key of Object.keys(setupEnvironment)) {
        if (key.toLowerCase() === "path" && key !== "PATH") {
          delete setupEnvironment[key];
        }
      }
      this.update({
        state: "installing",
        message:
          "Installing osAi CLI and compiling llama.cpp for this computer",
        percent: 70,
      });
      const setup = await runCommand(
        python.executable,
        [
          ...python.prefix,
          path.join(source, "scripts", "setup_osai.py"),
          "--venv",
          venv,
          "--skip-mlx-build",
          "--jobs",
          String(
            Math.max(
              1,
              Math.min(
                4,
                Math.floor(os.totalmem() / (8 * 1024 ** 3)),
                Math.floor(os.availableParallelism() / 2),
              ),
            ),
          ),
        ],
        {
          cwd: source,
          env: setupEnvironment,
          onLine: (line) => {
            const cleaned = cleanSetupLine(line);
            log = (log + `${line}\n`).slice(-2_000_000);
            const progress = cleaned.match(/^\[(\d+)\/(\d+)\]/);
            const percent = progress
              ? 70 +
                Math.floor((27 * Number(progress[1])) / Number(progress[2]))
              : undefined;
            if (
              Date.now() - lastStatusAt > 250 ||
              /error|failed/i.test(cleaned)
            ) {
              lastStatusAt = Date.now();
              this.update({ state: "installing", message: cleaned, percent });
            }
          },
        },
      );
      await fs.writeFile(installLog, log, { mode: 0o600 });
      if (setup.code !== 0)
        throw new Error(
          backendSetupFailureMessage(
            log,
            setup.code,
            path.join(this.installationsRoot, "last-install.log"),
          ),
        );

      const executable = backendExecutablePath(venv);
      const stat = await fs.stat(executable).catch(() => null);
      if (!stat?.isFile())
        throw new Error("Setup completed without creating the osAi executable");
      const check = await runCommand(executable, ["--version"], {
        timeoutMs: 20_000,
      });
      const version = osAiVersionFromOutput(`${check.stdout}\n${check.stderr}`);
      if (check.code !== 0 || !version)
        throw new Error(
          check.stderr.trim() ||
            "The installed executable did not identify itself as the osAi CLI",
        );
      if (
        version.toLowerCase() !==
        `osai ${sourceManifest.revision}`.toLowerCase()
      )
        throw new Error(
          `Downloaded CLI version ${version} does not match expected ${sourceManifest.revision}`,
        );
      await fs.writeFile(
        path.join(installRoot, "OSAI_BACKEND_SOURCE.json"),
        JSON.stringify({ ...sourceManifest, sha256: digest }, null, 2) + "\n",
        { mode: 0o600 },
      );
      await this.selectExecutable(executable);
      this.update({
        state: "ready",
        message: `${version} is ready`,
        percent: 100,
        executable,
        sourceDirectory: source,
      });
      return this.getStatus();
    } catch (error) {
      if (log) {
        await fs
          .mkdir(this.installationsRoot, { recursive: true, mode: 0o700 })
          .then(() =>
            fs.writeFile(
              path.join(this.installationsRoot, "last-install.log"),
              log,
              {
                mode: 0o600,
              },
            ),
          )
          .catch(() => undefined);
      }
      await fs
        .rm(installRoot, { recursive: true, force: true })
        .catch(() => undefined);
      this.update({
        state: "error",
        message: error instanceof Error ? error.message : String(error),
      });
      return this.getStatus();
    }
  }
}
