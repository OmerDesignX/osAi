#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(import.meta.dirname, "../..");

export function parsePnpmSpec(value) {
  if (!/^pnpm@\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(value ?? "")) {
    throw new Error(
      'package.json must declare an exact packageManager such as "pnpm@11.19.0"',
    );
  }
  return value;
}

export function pinnedPnpmSpec(projectRoot = root) {
  const packageJson = JSON.parse(
    fs.readFileSync(path.join(projectRoot, "package.json"), "utf8"),
  );
  return parsePnpmSpec(packageJson.packageManager);
}

export function pnpmCandidates(
  spec = pinnedPnpmSpec(),
  platform = process.platform,
) {
  const windows = platform === "win32";
  return [
    {
      label: "pnpm on PATH",
      command: windows ? "pnpm.cmd" : "pnpm",
      prefix: [],
      shell: windows,
    },
    {
      label: "Corepack",
      command: windows ? "corepack.cmd" : "corepack",
      prefix: ["pnpm"],
      shell: windows,
    },
    {
      label: `npm bootstrap (${spec})`,
      command: windows ? "npm.cmd" : "npm",
      prefix: ["exec", "--yes", `--package=${spec}`, "--", "pnpm"],
      shell: windows,
      env: {
        npm_config_cache: path.join(os.tmpdir(), "osai-release-pnpm-cache"),
      },
    },
  ];
}

function isAvailable(candidate) {
  const result = spawnSync(
    candidate.command,
    [...candidate.prefix, "--version"],
    {
      cwd: root,
      env: { ...process.env, ...(candidate.env ?? {}) },
      shell: candidate.shell,
      stdio: "ignore",
      windowsHide: true,
    },
  );
  return !result.error && result.status === 0;
}

let selectedCandidate;

export function resolvePnpm() {
  if (selectedCandidate) return selectedCandidate;

  const override = process.env.OSAI_PNPM_BIN
    ? {
        label: "OSAI_PNPM_BIN",
        command: process.env.OSAI_PNPM_BIN,
        prefix: [],
        shell:
          process.platform === "win32" &&
          /\.(?:cmd|bat)$/i.test(process.env.OSAI_PNPM_BIN),
      }
    : undefined;
  const candidates = [override, ...pnpmCandidates()].filter(Boolean);
  selectedCandidate = candidates.find(isAvailable);
  if (!selectedCandidate) {
    throw new Error(
      "Unable to start the package-pinned pnpm. Install Node.js 22 or newer with npm included, then retry.",
    );
  }

  if (selectedCandidate.label !== "pnpm on PATH") {
    process.stdout.write(
      `pnpm is not on PATH; using ${selectedCandidate.label}.\n`,
    );
  }
  return selectedCandidate;
}

export function runPnpm(args, options = {}) {
  const candidate = resolvePnpm();
  return new Promise((resolve, reject) => {
    process.stdout.write(`\n> pnpm ${args.join(" ")}\n`);
    const child = spawn(candidate.command, [...candidate.prefix, ...args], {
      cwd: options.cwd ?? root,
      env: {
        ...process.env,
        ...(candidate.env ?? {}),
        ...(options.env ?? {}),
      },
      shell: candidate.shell,
      stdio: "inherit",
      windowsHide: true,
    });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) resolve();
      else {
        reject(
          new Error(
            `pnpm exited with ${signal ? `signal ${signal}` : `code ${code}`}`,
          ),
        );
      }
    });
  });
}

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
if (invokedPath === fileURLToPath(import.meta.url)) {
  runPnpm(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`osAi release: ${error.message}\n`);
    process.exitCode = 1;
  });
}
