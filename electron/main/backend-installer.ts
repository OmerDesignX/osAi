import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import type { BackendInstallStatus } from "../types.js";

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

export function backendBundleTarget(
  platform = process.platform,
  architecture = process.arch,
) {
  const system =
    platform === "darwin"
      ? "macos"
      : platform === "win32"
        ? "windows"
        : platform === "linux"
          ? "linux"
          : platform;
  return `${system}-${architecture}`;
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
    private readonly backendBundleRoot: string,
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
    const copiedSource = path.join(installRoot, "source");
    const venv = path.join(installRoot, ".venv");
    let log = "";
    try {
      this.update({
        state: "preparing-python",
        message: "Checking the bundled Python 3.12 runtime",
      });
      const python = await verifyBundledPython(this.pythonRuntimeRoot);
      const bundleManifestPath = path.join(
        this.backendBundleRoot,
        "OSAI_BACKEND_BUNDLE.json",
      );
      const bundleManifest = JSON.parse(
        await fs.readFile(bundleManifestPath, "utf8").catch(() => {
          throw new Error(
            "This osAi App build does not contain its local training backend",
          );
        }),
      ) as {
        target?: string;
        requiredLlamaTargets?: string[];
      };
      const expectedTarget = backendBundleTarget();
      if (bundleManifest.target !== expectedTarget)
        throw new Error(
          `The bundled training backend targets ${bundleManifest.target || "an unknown platform"}, not ${expectedTarget}`,
        );
      const bundledSource = await findSourceRoot(
        path.join(this.backendBundleRoot, "source"),
      );
      const wheelhouse = path.join(this.backendBundleRoot, "wheelhouse");
      const wheels = await fs.readdir(wheelhouse).catch(() => []);
      if (!wheels.some((name) => name.endsWith(".whl")))
        throw new Error("This osAi App build has no offline Python packages");
      const suffix = process.platform === "win32" ? ".exe" : "";
      for (const target of bundleManifest.requiredLlamaTargets || []) {
        const candidates = [
          path.join(
            bundledSource,
            "vendor",
            "llama.cpp",
            "build",
            "bin",
            `${target}${suffix}`,
          ),
          path.join(
            bundledSource,
            "vendor",
            "llama.cpp",
            "build",
            "bin",
            "Release",
            `${target}${suffix}`,
          ),
        ];
        if (
          !(
            await Promise.all(
              candidates.map((file) => fs.stat(file).catch(() => null)),
            )
          ).some((details) => details?.isFile())
        )
          throw new Error(`The bundled llama.cpp target is missing: ${target}`);
      }
      await fs.mkdir(installRoot, { recursive: true, mode: 0o700 });
      this.update({
        state: "extracting",
        message: `Preparing the bundled osAi backend · Python ${python.version}`,
        percent: 10,
      });
      await fs.cp(bundledSource, copiedSource, {
        recursive: true,
        verbatimSymlinks: true,
      });
      const source = await findSourceRoot(copiedSource);

      const installLog = path.join(installRoot, "install.log");
      const setupEnvironment = {
        ...process.env,
        DO_NOT_TRACK: "1",
        HF_HUB_DISABLE_TELEMETRY: "1",
        PIP_DISABLE_PIP_VERSION_CHECK: "1",
        PIP_NO_INDEX: "1",
        PIP_NO_INPUT: "1",
        TOKENIZERS_PARALLELISM: "false",
        WANDB_MODE: "disabled",
      };
      this.update({
        state: "installing",
        message: "Installing the bundled Python packages and local engines",
        percent: 35,
      });
      const setup = await runCommand(
        python.executable,
        [
          ...python.prefix,
          path.join(source, "scripts", "setup_osai.py"),
          "--venv",
          venv,
          "--offline",
          "--wheelhouse",
          wheelhouse,
          "--skip-mlx-build",
          "--skip-llama-build",
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
