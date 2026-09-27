import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import path from "node:path";
import test from "node:test";

import {
  sessionTrainerPids,
  terminateProcessTree,
} from "../dist-electron/main/session-processes.js";

test("matches only trainer processes using the exact session directory", () => {
  const directory = path.resolve("C:/Users/oa/osAi/sessions/target");
  const other = path.resolve("C:/Users/oa/osAi/sessions/target-extra");
  const entries = [
    {
      pid: 111,
      name: "llama-finetune.exe",
      commandLine: `llama-finetune.exe -f "${directory}\\train-gpu-0.txt"`,
    },
    {
      pid: 222,
      name: "llama-finetune.exe",
      commandLine: `llama-finetune.exe -f "${other}\\train-gpu-0.txt"`,
    },
    {
      pid: 333,
      name: "llama-finetune.exe",
      commandLine: "llama-finetune.exe -f C:\\other\\train.txt",
    },
    {
      pid: 444,
      name: "not-a-trainer.exe",
      commandLine: `not-a-trainer.exe -f "${directory}\\train.txt"`,
    },
  ];
  assert.deepEqual(sessionTrainerPids(entries, directory, "win32"), [111]);
});

test("stops a selected subprocess without ending the controller", async () => {
  const child = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], {
    stdio: "ignore",
  });
  try {
    await new Promise((resolve, reject) => {
      child.once("spawn", resolve);
      child.once("error", reject);
    });
    await terminateProcessTree(child.pid);
    if (child.exitCode === null && child.signalCode === null) {
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(new Error("The selected subprocess did not exit")),
          2_000,
        );
        child.once("exit", () => {
          clearTimeout(timeout);
          resolve();
        });
      });
    }
    assert.ok(child.exitCode !== null || child.signalCode !== null);
  } finally {
    if (child.exitCode === null) child.kill("SIGKILL");
  }
});
