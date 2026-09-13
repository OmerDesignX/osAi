import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  inspectDataset,
  saveDataset,
} from "../dist-electron/main/dataset-editor.js";

async function writeRows(file, rows) {
  await fs.writeFile(
    file,
    rows.map((row) => JSON.stringify(row)).join("\n") + "\n",
    "utf8",
  );
}

test("inspects common chat, QA, translation, code, and preference layouts", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-data-inspect-"));
  try {
    const source = path.join(root, "mixed.jsonl");
    await writeRows(source, [
      {
        messages: [
          { role: "user", content: "Hi" },
          { role: "assistant", content: "Hello" },
        ],
      },
      { question: "Capital?", context: "France", answers: { text: ["Paris"] } },
      { translation: { en: "hello", fr: "bonjour" } },
      { description: "Add two values", code: "return a + b" },
      { prompt: "Pick", chosen: "Good", rejected: "Bad" },
      { mystery: true },
    ]);
    const inspection = await inspectDataset(source);
    assert.equal(inspection.totalRows, 6);
    assert.equal(inspection.validRows, 5);
    assert.equal(inspection.invalidRows, 1);
    assert.ok(inspection.formats.includes("chat"));
    assert.ok(inspection.formats.includes("preference"));
    assert.ok(inspection.recommendedTokenLimit >= 64);
    assert.match(inspection.issues[0].message, /Could not find/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("maps unfamiliar columns and exports a canonical deduplicated split dataset", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "osai-data-save-"));
  try {
    const source = path.join(root, "custom.jsonl");
    const output = path.join(root, "clean");
    await writeRows(source, [
      { ask: "One?", reply: "First" },
      { ask: "One?", reply: "First" },
      { ask: "Two?", reply: "Second" },
    ]);
    const inspection = await inspectDataset(source);
    assert.equal(inspection.invalidRows, 3);
    const result = await saveDataset({
      source,
      outputDirectory: output,
      task: "supervised",
      mapping: {
        messages: "",
        prompt: "ask",
        context: "",
        response: "reply",
        chosen: "",
        rejected: "",
        reward: "",
        text: "",
      },
      trimWhitespace: true,
      removeDuplicates: true,
      skipInvalid: false,
      validationPercent: 0,
      testPercent: 0,
      edits: [],
    });
    assert.equal(result.writtenRows, 2);
    assert.equal(result.duplicateRows, 1);
    assert.equal(result.inspection.invalidRows, 0);
    const rows = (await fs.readFile(path.join(output, "train.jsonl"), "utf8"))
      .trim()
      .split("\n")
      .map(JSON.parse);
    assert.equal(rows[0].messages[0].content, "One?");
    assert.equal(rows[0].messages[1].content, "First");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("the Data Editor is exposed through the safe bridge and session tabs", async () => {
  const [app, preload, main, styles] = await Promise.all([
    fs.readFile("src/App.tsx", "utf8"),
    fs.readFile("electron/preload/index.cts", "utf8"),
    fs.readFile("electron/main/index.ts", "utf8"),
    fs.readFile("src/styles.css", "utf8"),
  ]);
  assert.match(app, /<DataEditor/);
  assert.match(app, /<b>Data editor<\/b>/);
  assert.match(app, /Inspect or repair training data/);
  assert.match(preload, /inspectDataset:[\s\S]*?dataset:inspect/);
  assert.match(preload, /saveDataset:[\s\S]*?dataset:save/);
  assert.match(main, /ipcMain\.handle\("dataset:inspect"/);
  assert.match(main, /ipcMain\.handle\("dataset:save"/);
  assert.match(styles, /\.data-editor-grid\s*\{/);
});
