import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";

import { trustedBackendArchiveUrl } from "../dist-electron/main/backend-installer.js";

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

test("the app packages a CLI download manifest without backend binaries", async () => {
  const root = path.resolve(import.meta.dirname, "..");
  const app = JSON.parse(await fs.readFile(path.join(root, "package.json")));
  assert.equal(
    app.build.extraResources.some(
      (entry) =>
        entry.from === "releaseScripts/backend-source.json" &&
        entry.to === "backend-source.json",
    ),
    true,
  );
  const configuration = JSON.parse(
    await fs.readFile(path.join(root, "releaseScripts", "backend-source.json")),
  );
  assert.equal(configuration.ref, "main");
  assert.equal(configuration.revision, "0.1.4");
  assert.equal(
    configuration.archive,
    `https://codeload.github.com/OmerDesignX/osAi-CLI/zip/refs/heads/${configuration.ref}`,
  );
  assert.equal(trustedBackendArchiveUrl(configuration.archive), true);
  assert.equal(
    app.build.extraResources.some((entry) => entry.to === "backend"),
    false,
  );
});
