import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  backendInstallationIsCurrent,
  backendExecutablePath,
  bundledPythonExecutable,
  findSourceRoot,
} from "../dist-electron/main/backend-installer.js";

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

test("finds a complete downloaded CLI repository", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-installer-"));
  const repository = path.join(root, "osAi-CLI-main");
  try {
    await fs.mkdir(path.join(repository, "scripts"), { recursive: true });
    await fs.mkdir(path.join(repository, "vendor", "llama.cpp"), {
      recursive: true,
    });
    await fs.writeFile(path.join(repository, "pyproject.toml"), "[project]\n");
    await fs.writeFile(path.join(repository, "scripts", "setup_osai.py"), "");
    await fs.writeFile(
      path.join(repository, "vendor", "llama.cpp", "CMakeLists.txt"),
      "",
    );
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

test("managed backends update when the CLI download reference changes", async () => {
  const root = await fs.mkdtemp(
    path.join(os.tmpdir(), "osai-backend-version-"),
  );
  const installations = path.join(root, "installations");
  const sourceManifest = path.join(root, "backend-source.json");
  const install = path.join(installations, "20260908");
  const executable = backendExecutablePath(path.join(install, ".venv"));
  const packagedManifest = {
    repository: "https://github.com/OmerDesignX/osAi-CLI",
    archive:
      "https://codeload.github.com/OmerDesignX/osAi-CLI/zip/refs/heads/main",
    ref: "main",
  };
  try {
    await fs.mkdir(path.dirname(executable), { recursive: true });
    await fs.writeFile(executable, "");
    await fs.writeFile(sourceManifest, JSON.stringify(packagedManifest));

    assert.equal(
      await backendInstallationIsCurrent(
        executable,
        installations,
        sourceManifest,
      ),
      false,
    );
    await fs.writeFile(
      path.join(install, "OSAI_BACKEND_SOURCE.json"),
      JSON.stringify({ ...packagedManifest, ref: "old-source" }),
    );
    assert.equal(
      await backendInstallationIsCurrent(
        executable,
        installations,
        sourceManifest,
      ),
      false,
    );
    await fs.writeFile(
      path.join(install, "OSAI_BACKEND_SOURCE.json"),
      JSON.stringify(packagedManifest),
    );
    assert.equal(
      await backendInstallationIsCurrent(
        executable,
        installations,
        sourceManifest,
      ),
      true,
    );
    assert.equal(
      await backendInstallationIsCurrent(
        path.join(root, "external", "osai"),
        installations,
        sourceManifest,
      ),
      true,
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
