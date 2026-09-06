import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import type { BackendInstallStatus } from "../types.js";

const BACKEND_ARCHIVE_URL =
  "https://codeload.github.com/OmerDesignX/osAi-CLI/zip/refs/heads/main";
const MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024;
const PYTHON_CHECK =
  "import sys; print('.'.join(map(str, sys.version_info[:3]))); " +
  "raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)";

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

export function isTrustedBackendSourceUrl(raw: string) {
  try {
    const url = new URL(raw);
    return (
      url.protocol === "https:" &&
      (url.hostname === "codeload.github.com" ||
        url.hostname === "github.com" ||
        url.hostname === "objects.githubusercontent.com")
    );
  } catch {
    return false;
  }
}

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

export async function findSourceRoot(extractedRoot: string) {
  const candidates = [
    extractedRoot,
    ...(await fs
      .readdir(extractedRoot, { withFileTypes: true })
      .then((entries) =>
        entries
          .filter((entry) => entry.isDirectory())
          .map((entry) => path.join(extractedRoot, entry.name)),
      )),
  ];
  for (const candidate of candidates) {
    const required = [
      path.join(candidate, "pyproject.toml"),
      path.join(candidate, "scripts", "setup_osai.py"),
      path.join(candidate, "vendor", "llama.cpp"),
    ];
    if (
      (
        await Promise.all(
          required.map((file) => fs.stat(file).catch(() => null)),
        )
      ).every(Boolean)
    )
      return candidate;
  }
  throw new Error(
    "The downloaded archive is not a complete osAi CLI repository",
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

async function downloadArchive(
  destination: string,
  update: (downloaded: number, total?: number) => void,
) {
  const response = await fetch(BACKEND_ARCHIVE_URL, {
    redirect: "follow",
    headers: {
      Accept: "application/zip",
      "User-Agent": "osAi-App/0.1.0",
    },
  });
  if (!response.ok || !response.body)
    throw new Error(`Could not download osAi CLI (${response.status})`);
  if (!isTrustedBackendSourceUrl(response.url))
    throw new Error("The osAi CLI download redirected to an untrusted server");
  const length = Number(response.headers.get("content-length"));
  const total = Number.isFinite(length) && length > 0 ? length : undefined;
  if (total && total > MAX_ARCHIVE_BYTES)
    throw new Error(
      "The osAi CLI archive is larger than the safe download limit",
    );

  const handle = await fs.open(destination, "wx", 0o600);
  const reader = response.body.getReader();
  let downloaded = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      downloaded += value.byteLength;
      if (downloaded > MAX_ARCHIVE_BYTES)
        throw new Error(
          "The osAi CLI archive exceeded the safe download limit",
        );
      let offset = 0;
      while (offset < value.byteLength) {
        const { bytesWritten } = await handle.write(
          value,
          offset,
          value.byteLength - offset,
        );
        if (bytesWritten < 1)
          throw new Error("Could not write the osAi CLI archive");
        offset += bytesWritten;
      }
      update(downloaded, total);
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    await handle.close();
  }
  if (downloaded < 1) throw new Error("The osAi CLI download was empty");
}

function cleanSetupLine(line: string) {
  return line.replace(/\s+/g, " ").trim().slice(0, 220);
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
    private readonly emit: (status: BackendInstallStatus) => void,
    private readonly selectExecutable: (executable: string) => Promise<void>,
  ) {}

  getStatus() {
    return { ...this.status };
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
    const extracted = path.join(installRoot, "source");
    const venv = path.join(installRoot, ".venv");
    let log = "";
    try {
      this.update({
        state: "preparing-python",
        message: "Checking the bundled Python 3.12 runtime",
      });
      const python = await verifyBundledPython(this.pythonRuntimeRoot);
      await fs.mkdir(extracted, { recursive: true, mode: 0o700 });
      this.update({
        state: "downloading",
        message: `Downloading the complete osAi CLI repository · Python ${python.version}`,
        percent: 0,
      });
      let lastReported = 0;
      await downloadArchive(archive, (downloaded, total) => {
        const now = Date.now();
        if (now - lastReported < 150 && total && downloaded < total) return;
        lastReported = now;
        const percent = total
          ? Math.min(100, Math.floor((downloaded / total) * 100))
          : undefined;
        const mib = (downloaded / 1024 / 1024).toFixed(1);
        this.update({
          state: "downloading",
          message: `Downloading the complete osAi CLI repository · ${mib} MiB`,
          percent,
        });
      });

      this.update({
        state: "extracting",
        message: "Extracting the local osAi CLI repository",
      });
      const extract = await runCommand(
        python.executable,
        [...python.prefix, "-m", "zipfile", "-e", archive, extracted],
        { timeoutMs: 20 * 60_000 },
      );
      if (extract.code !== 0)
        throw new Error(
          extract.stderr.trim() || "Could not extract the osAi CLI archive",
        );
      const source = await findSourceRoot(extracted);
      await fs.rm(archive, { force: true });

      const installLog = path.join(installRoot, "install.log");
      const setupEnvironment = {
        ...process.env,
        DO_NOT_TRACK: "1",
        HF_HUB_DISABLE_TELEMETRY: "1",
        PIP_DISABLE_PIP_VERSION_CHECK: "1",
        TOKENIZERS_PARALLELISM: "false",
        WANDB_MODE: "disabled",
      };
      this.update({
        state: "installing",
        message: "Installing Python packages and building the local engines",
      });
      const setup = await runCommand(
        python.executable,
        [
          ...python.prefix,
          path.join(source, "scripts", "setup_osai.py"),
          "--venv",
          venv,
        ],
        {
          cwd: source,
          env: setupEnvironment,
          onLine: (line) => {
            const cleaned = cleanSetupLine(line);
            log += `${cleaned}\n`;
            this.update({ state: "installing", message: cleaned });
          },
        },
      );
      await fs.writeFile(installLog, log, { mode: 0o600 });
      if (setup.code !== 0)
        throw new Error(
          cleanSetupLine(
            setup.stderr.trim().split(/\r?\n/).filter(Boolean).at(-1) ||
              "osAi setup failed",
          ),
        );

      const executable = backendExecutablePath(venv);
      const stat = await fs.stat(executable).catch(() => null);
      if (!stat?.isFile())
        throw new Error("Setup completed without creating the osAi executable");
      const check = await runCommand(executable, ["--version"], {
        timeoutMs: 20_000,
      });
      if (check.code !== 0)
        throw new Error(
          check.stderr.trim() ||
            "The installed osAi CLI did not start correctly",
        );
      await this.selectExecutable(executable);
      this.update({
        state: "ready",
        message: `osAi ${check.stdout.trim() || "CLI"} is ready`,
        percent: 100,
        executable,
        sourceDirectory: source,
      });
      return this.getStatus();
    } catch (error) {
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
