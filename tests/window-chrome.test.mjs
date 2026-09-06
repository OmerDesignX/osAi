import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("macOS uses a dedicated draggable title-bar handle", async () => {
  const root = path.resolve(import.meta.dirname, "..");
  const app = await fs.readFile(path.join(root, "src", "App.tsx"), "utf8");
  const styles = await fs.readFile(
    path.join(root, "src", "styles.css"),
    "utf8",
  );
  assert.match(app, /className="mac-titlebar-safe-area"/);
  assert.match(
    styles,
    /\.platform-darwin \.mac-titlebar-safe-area\s*\{[\s\S]*?height: 30px;[\s\S]*?-webkit-app-region: drag;/,
  );
  assert.match(
    styles,
    /\.app\.platform-darwin\s*\{\s*grid-template-rows: 30px 68px minmax\(0, 1fr\) 46px;/,
  );
});
