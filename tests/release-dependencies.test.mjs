import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

async function javascriptFiles(directory) {
  const files = [];
  for (const item of await fs.readdir(directory, { withFileTypes: true })) {
    const name = path.join(directory, item.name);
    if (item.isDirectory()) files.push(...(await javascriptFiles(name)));
    else if (item.isFile() && /\.[cm]?js$/.test(item.name)) files.push(name);
  }
  return files;
}

test("packaged Electron code needs no production node modules", async () => {
  const manifest = JSON.parse(
    await fs.readFile(path.join(root, "package.json"), "utf8"),
  );
  assert.deepEqual(manifest.dependencies, {});
  for (const file of await javascriptFiles(path.join(root, "dist-electron"))) {
    const source = await fs.readFile(file, "utf8");
    const imports = source.matchAll(
      /(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*)["']([^"']+)["']/g,
    );
    for (const [, specifier] of imports) {
      assert.ok(
        specifier.startsWith(".") ||
          specifier.startsWith("node:") ||
          specifier === "electron",
        `${path.relative(root, file)} imports production module ${specifier}`,
      );
    }
  }
});
