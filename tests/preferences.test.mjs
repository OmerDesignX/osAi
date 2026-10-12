import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  defaultSessionsRoot,
  readPreferences,
  writePreferences,
} from "../dist-electron/main/preferences.js";

test("defaults sessions to an osAi folder in the user home directory", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-preferences-"));
  try {
    const preferences = await readPreferences(path.join(root, "missing.json"));
    assert.equal(
      defaultSessionsRoot,
      path.join(os.homedir(), "osAi", "sessions"),
    );
    assert.equal(preferences.sessionsRoot, defaultSessionsRoot);
    assert.deepEqual(preferences.sessionRoots, [defaultSessionsRoot]);
    assert.equal(preferences.clusterMode, "off");
    assert.equal(preferences.clusterSecret, "");
    assert.ok(preferences.clusterName.length > 0);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("validates and bounds private cluster preferences", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-preferences-"));
  const file = path.join(root, "preferences.json");
  try {
    const saved = await writePreferences(file, {
      version: 1,
      sessionsRoot: defaultSessionsRoot,
      clusterMode: "master",
      clusterName: `  ${"machine".repeat(20)}  `,
      clusterSecret: `  ${"s".repeat(300)}  `,
    });
    assert.equal(saved.clusterMode, "master");
    assert.equal(saved.clusterName.length, 80);
    assert.equal(saved.clusterSecret.length, 256);

    const rejected = await writePreferences(file, {
      version: 1,
      sessionsRoot: defaultSessionsRoot,
      clusterMode: "public-internet",
      clusterName: "",
      clusterSecret: 42,
    });
    assert.equal(rejected.clusterMode, "off");
    assert.equal(rejected.clusterSecret, "");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("remembers prior custom roots when the save location changes", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-preferences-"));
  const file = path.join(root, "preferences.json");
  const previous = path.join(root, "previous");
  const current = path.join(root, "current");
  try {
    const saved = await writePreferences(file, {
      version: 1,
      theme: "blue-dark",
      backendExecutable: "osai",
      autoUpdateEnabled: false,
      sessionsRoot: current,
      sessionRoots: [previous],
    });
    assert.equal(saved.sessionsRoot, current);
    assert.deepEqual(saved.sessionRoots, [current, previous]);
    assert.deepEqual(await readPreferences(file), saved);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
