import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { migrateLegacyV1Models } from "../dist-electron/main/model-migration.js";

async function fixture(root, format, tier = "small") {
  const directory =
    format === "MLX"
      ? path.join(root, "MLX", "osCode-MLX-Small-Q5")
      : path.join(root, "GGUF", tier);
  const required =
    format === "MLX"
      ? [
          "config.json",
          "model.safetensors.index.json",
          "tokenizer.json",
          "tokenizer_config.json",
          ...Array.from(
            { length: 21 },
            (_, index) =>
              `model-${String(index + 1).padStart(5, "0")}-of-00021.safetensors`,
          ),
        ]
      : [
          "osCode-GGUF-Small-Q4_K_M-00001-of-00002.gguf",
          "osCode-GGUF-Small-Q4_K_M-00002-of-00002.gguf",
        ];
  await fs.mkdir(directory, { recursive: true });
  const files = [];
  for (const name of required) {
    const content = Buffer.from(`fixture ${name}`);
    await fs.writeFile(path.join(directory, name), content);
    files.push({
      path: name,
      bytes: content.length,
      sha256: createHash("sha256").update(content).digest("hex"),
    });
  }
  const manifest = {
    schema_version: 1,
    repository: "https://github.com/OmerDesignX/osCode-Models",
    repository_path:
      format === "MLX" ? "MLX/osCode-MLX-Small-Q5" : `GGUF/${required[0]}`,
    runtime: format === "MLX" ? "mlx" : "llama.cpp",
    tier,
    files,
  };
  await fs.writeFile(
    path.join(directory, "OSCODE_MODEL.json"),
    JSON.stringify(manifest),
  );
  return { directory, manifest };
}

test("an upgrade promotes complete V1 models while preserving older session paths", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-v1-migration-"));
  try {
    const mlx = await fixture(root, "MLX");
    const gguf = await fixture(root, "GGUF");
    const v2 = path.join(root, "V2", "MLX", "osCode-MLX-Small-Q5");
    await fs.mkdir(v2, { recursive: true });
    await fs.writeFile(path.join(v2, "sentinel"), "V2 is untouched");

    assert.deepEqual(await migrateLegacyV1Models(root, true), []);
    assert.ok((await fs.stat(mlx.directory)).isDirectory());
    assert.deepEqual(await migrateLegacyV1Models(root), [
      path.join("MLX", "osCode-MLX-Small-Q5"),
      path.join("GGUF", "small"),
    ]);
    assert.ok((await fs.stat(mlx.directory)).isDirectory());
    assert.ok((await fs.stat(gguf.directory)).isDirectory());
    assert.ok(
      (
        await fs.stat(
          path.join(
            root,
            "V1",
            "MLX",
            "osCode-MLX-Small-Q5",
            "OSCODE_MODEL.json",
          ),
        )
      ).isFile(),
    );
    assert.ok(
      (
        await fs.stat(
          path.join(root, "V1", "GGUF", "small", gguf.manifest.files[1].path),
        )
      ).isFile(),
    );
    assert.equal(
      await fs.readFile(path.join(v2, "sentinel"), "utf8"),
      "V2 is untouched",
    );
    assert.deepEqual(await migrateLegacyV1Models(root), []);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("migration preserves incomplete models and never overwrites an existing V1", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-v1-migration-"));
  try {
    const mlx = await fixture(root, "MLX");
    const gguf = await fixture(root, "GGUF");
    await fs.unlink(path.join(mlx.directory, mlx.manifest.files.at(-1).path));
    const destination = path.join(root, "V1", "GGUF", "small");
    await fs.mkdir(destination, { recursive: true });
    await fs.writeFile(path.join(destination, "sentinel"), "existing model");

    assert.deepEqual(await migrateLegacyV1Models(root), []);
    assert.ok((await fs.stat(gguf.directory)).isDirectory());
    assert.equal(
      await fs.readFile(path.join(destination, "sentinel"), "utf8"),
      "existing model",
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("migration refuses a V1 destination symlink", async (context) => {
  if (process.platform === "win32") {
    context.skip(
      "directory symlinks may require special permissions on Windows",
    );
    return;
  }
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-v1-migration-"));
  const outside = await fs.mkdtemp(path.join(os.tmpdir(), "osai-v1-outside-"));
  try {
    const mlx = await fixture(root, "MLX");
    await fs.symlink(outside, path.join(root, "V1"), "dir");
    assert.deepEqual(await migrateLegacyV1Models(root), []);
    assert.ok((await fs.stat(mlx.directory)).isDirectory());
    assert.deepEqual(await fs.readdir(outside), []);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
    await fs.rm(outside, { recursive: true, force: true });
  }
});
