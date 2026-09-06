import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";

import {
  bundleTarget,
  releaseBuildJobs,
  trustedBackendArchiveUrl,
} from "../scripts/prepare-backend-bundle.mjs";

test("backend release downloads accept only the pinned GitHub archive host", () => {
  assert.equal(
    trustedBackendArchiveUrl(
      "https://codeload.github.com/OmerDesignX/osAi-CLI/zip/refs/heads/main",
    ),
    true,
  );
  assert.equal(
    trustedBackendArchiveUrl("http://codeload.github.com/archive.zip"),
    false,
  );
  assert.equal(
    trustedBackendArchiveUrl("https://codeload.github.com.example/archive.zip"),
    false,
  );
});

test("native backend compilation is memory-capped by default", () => {
  assert.equal(releaseBuildJobs(8 * 1024 ** 3), 2);
  assert.equal(releaseBuildJobs(16 * 1024 ** 3), 4);
  assert.equal(releaseBuildJobs(2 * 1024 ** 3), 1);
});

test("backend bundle targets are explicit and platform-safe", () => {
  assert.equal(bundleTarget("macos", "arm64"), "macos-arm64");
  assert.equal(bundleTarget("windows", "x64"), "windows-x64");
  assert.throws(() => bundleTarget("windows", "arm64"), /x64 only/);
  assert.throws(() => bundleTarget("freebsd", "x64"), /Unsupported/);
});

test("the packaged app includes the self-contained backend bundle", async () => {
  const root = path.resolve(import.meta.dirname, "..");
  const app = JSON.parse(await fs.readFile(path.join(root, "package.json")));
  assert.equal(
    app.build.extraResources.some(
      (entry) =>
        entry.from === "build/backend-bundle" && entry.to === "backend",
    ),
    true,
  );
  const configuration = JSON.parse(
    await fs.readFile(path.join(root, "releaseScripts", "backend-bundle.json")),
  );
  assert.deepEqual(configuration.requiredLlamaTargets.sort(), [
    "llama-completion",
    "llama-export-lora",
    "llama-finetune",
    "llama-gguf-split",
    "llama-perplexity",
    "llama-quantize",
  ]);
});
