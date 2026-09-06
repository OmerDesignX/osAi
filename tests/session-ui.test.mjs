import assert from "node:assert/strict";
import fs from "node:fs/promises";
import test from "node:test";

test("session tabs expose a padded delete menu through the safe bridge", async () => {
  const [app, styles, preload, main] = await Promise.all([
    fs.readFile("src/App.tsx", "utf8"),
    fs.readFile("src/styles.css", "utf8"),
    fs.readFile("electron/preload/index.cts", "utf8"),
    fs.readFile("electron/main/index.ts", "utf8"),
  ]);
  assert.match(app, /className="session-tab-more"/);
  assert.match(app, /Delete session/);
  assert.match(styles, /\.session-tab-more\s*\{[^}]*padding:\s*0 8px 0 3px/s);
  assert.match(preload, /deleteSession:.*training:delete/);
  assert.match(main, /training:delete/);
  assert.match(main, /shell\.trashItem/);
});

test("notifications reveal complete cleaned error details", async () => {
  const app = await fs.readFile("src/App.tsx", "utf8");
  assert.match(app, /function readableError/);
  assert.match(app, /Error invoking remote method/);
  assert.match(app, /aria-label="Show complete error"/);
  assert.match(app, /className="app-dialog error-details-dialog"/);
  assert.match(app, /<pre>\{notice\}<\/pre>/);
});
