import assert from "node:assert/strict";
import test from "node:test";
import {
  isNewerVersion,
  isTrustedUpdateUrl,
  selectUpdateAsset,
  updateAssetName,
  updateChannel,
} from "../dist-electron/main/updater-policy.js";

test("selects platform-specific updater channels and packages", () => {
  assert.equal(
    updateChannel("darwin", "arm64")?.tag,
    "macOS-Apple-Silicon-Updater",
  );
  assert.equal(updateChannel("darwin", "x64")?.tag, "macOS-Intel-Updater");
  assert.equal(
    updateChannel("win32", "x64", "10.0.19045")?.tag,
    "Windows-10-Updater",
  );
  assert.equal(
    updateChannel("win32", "x64", "10.0.22631")?.tag,
    "Windows-11-Updater",
  );
  assert.equal(updateChannel("linux", "x64")?.tag, "Linux-Updater");
  assert.equal(
    updateAssetName("1.2.3", "darwin", "arm64"),
    "osAi-1.2.3-mac-arm64.dmg",
  );
  assert.equal(
    updateAssetName("1.2.3", "win32", "x64"),
    "osAi-Setup-1.2.3.exe",
  );
  assert.equal(updateAssetName("1.2.3", "linux", "x64"), "osAi-1.2.3-x64.deb");
});

test("rejects untrusted update URLs and selects only newer native assets", () => {
  assert.equal(
    isTrustedUpdateUrl("https://api.github.com/repos/OmerDesignX/osAi-CLI"),
    true,
  );
  assert.equal(
    isTrustedUpdateUrl("http://github.com/OmerDesignX/osAi-CLI"),
    false,
  );
  assert.equal(isTrustedUpdateUrl("https://example.com/update.dmg"), false);
  assert.equal(isNewerVersion("0.2.0", "0.1.9"), true);
  const selected = selectUpdateAsset(
    [
      { name: "osAi-0.1.0-mac-arm64.dmg" },
      { name: "osAi-0.3.0-mac-x64.dmg" },
      { name: "osAi-0.2.0-mac-arm64.dmg" },
    ],
    "0.1.0",
    "darwin",
    "arm64",
  );
  assert.equal(selected?.version, "0.2.0");
});
