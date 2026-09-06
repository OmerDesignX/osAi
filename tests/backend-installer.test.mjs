import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  backendBundleTarget,
  backendExecutablePath,
  bundledPythonExecutable,
  findSourceRoot,
} from "../dist-electron/main/backend-installer.js";

test("maps Electron platforms to packaged backend targets", () => {
  assert.equal(backendBundleTarget("darwin", "arm64"), "macos-arm64");
  assert.equal(backendBundleTarget("darwin", "x64"), "macos-x64");
  assert.equal(backendBundleTarget("win32", "x64"), "windows-x64");
  assert.equal(backendBundleTarget("linux", "x64"), "linux-x64");
});

test("uses the native virtual-environment executable layout", () => {
  assert.equal(
    backendExecutablePath("/install/.venv", "darwin"),
    path.join("/install/.venv", "bin", "osai"),
  );
  assert.equal(
    backendExecutablePath("C:\\install\\.venv", "win32"),
    path.join("C:\\install\\.venv", "Scripts", "osai.exe"),
  );
});

test("uses only the Python runtime packaged inside the application", () => {
  assert.equal(
    bundledPythonExecutable("/app/resources/python", "darwin"),
    path.join("/app/resources/python", "bin", "python3"),
  );
  assert.equal(
    bundledPythonExecutable("C:\\app\\resources\\python", "win32"),
    path.join("C:\\app\\resources\\python", "python.exe"),
  );
});

test("finds a complete repository inside the packaged backend", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-installer-"));
  const repository = path.join(root, "osAi-CLI-main");
  try {
    await fs.mkdir(path.join(repository, "scripts"), { recursive: true });
    await fs.mkdir(path.join(repository, "vendor", "llama.cpp"), {
      recursive: true,
    });
    await fs.writeFile(path.join(repository, "pyproject.toml"), "[project]\n");
    await fs.writeFile(path.join(repository, "scripts", "setup_osai.py"), "");
    assert.equal(await findSourceRoot(root), repository);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("rejects an incomplete downloaded repository", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-installer-"));
  try {
    await assert.rejects(
      findSourceRoot(root),
      /not a complete osAi CLI repository/,
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
