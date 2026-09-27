import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  EXTRACT_ARCHIVE,
  backendInstallationIsCurrent,
  backendExecutablePath,
  bundledPythonExecutable,
  findSourceRoot,
} from "../dist-electron/main/backend-installer.js";
import {
  backendRuntimeEnvironment,
  managedBackendSource,
} from "../dist-electron/main/backend-source.js";

const testPython = [process.env.OSAI_TEST_PYTHON, "python3", "python"].find(
  (candidate) =>
    candidate &&
    spawnSync(candidate, ["-c", "import zipfile"], {
      stdio: "ignore",
    }).status === 0,
);

test(
  "extracts CLI build sources while omitting llama.cpp server UI paths",
  { skip: !testPython },
  async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-archive-"));
    const archive = path.join(root, "source.zip");
    const destination = path.join(
      root,
      "backend",
      "installations",
      "20260926201432394",
      "source",
    );
    const createArchive = `import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w") as archive:
    archive.writestr("osAi-CLI-main/pyproject.toml", "[project]\\n")
    archive.writestr("osAi-CLI-main/scripts/setup_osai.py", "")
    archive.writestr("osAi-CLI-main/vendor/llama.cpp/CMakeLists.txt", "")
    archive.writestr("osAi-CLI-main/vendor/llama.cpp/gguf-py/gguf/__init__.py", "")
    archive.writestr("osAi-CLI-main/vendor/llama.cpp/tools/ui/src/lib/components/app/chat/ChatAttachments/ChatAttachmentsPreview/ChatAttachmentsPreviewCurrentItem/ChatAttachmentsPreviewCurrentItemUnavailable.svelte", "unused")`;
    try {
      const created = spawnSync(testPython, ["-c", createArchive, archive], {
        encoding: "utf8",
      });
      assert.equal(created.status, 0, created.stderr);
      const extracted = spawnSync(
        testPython,
        ["-c", EXTRACT_ARCHIVE, archive, destination],
        { encoding: "utf8" },
      );
      assert.equal(extracted.status, 0, extracted.stderr);
      const source = await findSourceRoot(destination);
      assert.equal(
        (
          await fs.stat(
            path.join(
              source,
              "vendor",
              "llama.cpp",
              "gguf-py",
              "gguf",
              "__init__.py",
            ),
          )
        ).isFile(),
        true,
      );
      if (process.platform === "win32")
        assert.ok(
          path.join(
            source,
            "vendor",
            "llama.cpp",
            "tools",
            "ui",
            "src",
            "lib",
            "components",
            "app",
            "chat",
            "ChatAttachments",
            "ChatAttachmentsPreview",
            "ChatAttachmentsPreviewCurrentItem",
            "ChatAttachmentsPreviewCurrentItemUnavailable.svelte",
          ).length > 260,
        );
      await assert.rejects(
        fs.stat(path.join(source, "vendor", "llama.cpp", "tools", "ui")),
      );
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  },
);

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
  const downloadedSource = path.join(install, "source", "osAi-CLI-main");
  const packagedManifest = {
    repository: "https://github.com/OmerDesignX/osAi-CLI",
    archive:
      "https://codeload.github.com/OmerDesignX/osAi-CLI/zip/refs/heads/main",
    ref: "main",
  };
  try {
    await fs.mkdir(path.dirname(executable), { recursive: true });
    await fs.writeFile(executable, "");
    await fs.mkdir(path.join(downloadedSource, "scripts"), { recursive: true });
    await fs.mkdir(path.join(downloadedSource, "vendor", "llama.cpp"), {
      recursive: true,
    });
    await fs.writeFile(
      path.join(downloadedSource, "pyproject.toml"),
      "[project]\n",
    );
    await fs.writeFile(
      path.join(downloadedSource, "scripts", "setup_osai.py"),
      "",
    );
    await fs.writeFile(
      path.join(downloadedSource, "vendor", "llama.cpp", "CMakeLists.txt"),
      "",
    );
    await fs.writeFile(sourceManifest, JSON.stringify(packagedManifest));

    assert.equal(await managedBackendSource(executable), downloadedSource);
    const resources = path.join(root, "resources");
    const originalResources = Object.getOwnPropertyDescriptor(
      process,
      "resourcesPath",
    );
    Object.defineProperty(process, "resourcesPath", {
      configurable: true,
      value: resources,
    });
    let runtime;
    try {
      runtime = await backendRuntimeEnvironment(executable, {
        TEST_SETTING: "1",
        PATH: "system-tools",
      });
    } finally {
      if (originalResources)
        Object.defineProperty(process, "resourcesPath", originalResources);
      else delete process.resourcesPath;
    }
    assert.equal(runtime.OSAI_ROOT, downloadedSource);
    assert.equal(runtime.TEST_SETTING, "1");
    assert.equal(
      runtime.OSAI_CMAKE,
      path.join(
        resources,
        "native-tools",
        "cmake",
        "bin",
        process.platform === "win32" ? "cmake.exe" : "cmake",
      ),
    );
    assert.ok(
      runtime.PATH.startsWith(path.dirname(executable) + path.delimiter),
    );
    assert.ok(
      runtime.PATH.includes(path.join(resources, "native-tools", "bin")),
    );
    assert.ok(runtime.PATH.endsWith("system-tools"));

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
    await fs.rm(downloadedSource, { recursive: true, force: true });
    assert.equal(
      await backendInstallationIsCurrent(
        executable,
        installations,
        sourceManifest,
      ),
      false,
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
