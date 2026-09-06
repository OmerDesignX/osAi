import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import {
  runtimeExecutable,
  trustedPythonRuntimeUrl,
} from "../scripts/prepare-python-runtime.mjs";

test("the bundled runtime is pinned and included as an extra resource", async () => {
  const root = path.resolve(import.meta.dirname, "..");
  const runtime = JSON.parse(
    await fs.readFile(path.join(root, "releaseScripts", "python-runtime.json")),
  );
  const app = JSON.parse(await fs.readFile(path.join(root, "package.json")));
  assert.equal(runtime.pythonVersion, "3.12.14");
  assert.equal(runtime.release, "20260901");
  assert.deepEqual(Object.keys(runtime.runtimes).sort(), [
    "linux-x64",
    "macos-arm64",
    "macos-x64",
    "windows-x64",
  ]);
  assert.equal(
    app.build.extraResources.some(
      (entry) => entry.from === "build/python-runtime" && entry.to === "python",
    ),
    true,
  );
});

test("runtime downloads accept only trusted HTTPS release hosts", () => {
  assert.equal(
    trustedPythonRuntimeUrl(
      "https://github.com/astral-sh/python-build-standalone/releases/download/tag/python.tar.gz",
    ),
    true,
  );
  assert.equal(
    trustedPythonRuntimeUrl("http://github.com/python.tar.gz"),
    false,
  );
  assert.equal(
    trustedPythonRuntimeUrl("https://github.com.example/python.tar.gz"),
    false,
  );
});

test("runtime executable paths match each packaged operating system", () => {
  assert.equal(
    runtimeExecutable("/runtime", "macos"),
    path.join("/runtime", "bin", "python3"),
  );
  assert.equal(
    runtimeExecutable("C:\\runtime", "windows"),
    path.join("C:\\runtime", "python.exe"),
  );
});
