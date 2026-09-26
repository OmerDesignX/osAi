import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";

import {
  parsePnpmSpec,
  pinnedPnpmSpec,
  pnpmCandidates,
} from "../releaseScripts/common/run-pnpm.mjs";

const root = path.resolve(import.meta.dirname, "..");

test("release bootstrap uses the package-pinned pnpm version", () => {
  assert.equal(pinnedPnpmSpec(root), "pnpm@11.19.0");
  assert.equal(parsePnpmSpec("pnpm@11.19.0"), "pnpm@11.19.0");
  assert.throws(() => parsePnpmSpec("pnpm@latest"), /exact packageManager/);
});

test("release bootstrap provides native and npm fallbacks on every platform", () => {
  assert.deepEqual(
    pnpmCandidates("pnpm@11.19.0", "darwin").map(({ command, prefix }) => ({
      command,
      prefix,
    })),
    [
      { command: "pnpm", prefix: [] },
      { command: "corepack", prefix: ["pnpm"] },
      {
        command: "npm",
        prefix: ["exec", "--yes", "--package=pnpm@11.19.0", "--", "pnpm"],
      },
    ],
  );
  assert.equal(pnpmCandidates("pnpm@11.19.0", "win32")[2].command, "npm.cmd");
});

test("native release entry points use the shared pnpm bootstrap", async () => {
  const files = [
    "releaseScripts/macos/build.sh",
    "releaseScripts/linux/build.sh",
    "releaseScripts/windows/build-windows.sh",
    "releaseScripts/common/prepare-source.sh",
  ];
  for (const relative of files) {
    const source = await fs.readFile(path.join(root, relative), "utf8");
    assert.match(source, /run-pnpm\.mjs/, relative);
    if (relative === "releaseScripts/windows/build-windows.sh") {
      assert.match(source, /run-pnpm\.mjs" exec electron-builder --win/);
    }
  }
});
