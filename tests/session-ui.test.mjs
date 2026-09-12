import assert from "node:assert/strict";
import fs from "node:fs/promises";
import test from "node:test";

test("session tabs expose confirmed restart and delete actions through the safe bridge", async () => {
  const [app, styles, preload, main] = await Promise.all([
    fs.readFile("src/App.tsx", "utf8"),
    fs.readFile("src/styles.css", "utf8"),
    fs.readFile("electron/preload/index.cts", "utf8"),
    fs.readFile("electron/main/index.ts", "utf8"),
  ]);
  assert.match(app, /className="session-tab-more"/);
  assert.match(app, /Restart session/);
  assert.match(app, /Clear pipeline and restart\?/);
  assert.match(app, /Clear and restart/);
  assert.match(app, /Delete session/);
  assert.match(styles, /\.session-tab-more\s*\{[^}]*padding:\s*0 8px 0 3px/s);
  assert.match(preload, /deleteSession:.*training:delete/);
  assert.match(preload, /restartSession:[\s\S]*?training:restart/);
  assert.match(main, /training:delete/);
  assert.match(main, /training:restart/);
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

test("live output follows only at the bottom and offers a subtle latest control", async () => {
  const [app, styles] = await Promise.all([
    fs.readFile("src/App.tsx", "utf8"),
    fs.readFile("src/styles.css", "utf8"),
  ]);
  assert.match(app, /followLogRef\.current = atBottom/);
  assert.match(app, /distanceFromBottom <= 24/);
  assert.match(app, /aria-label="Jump to latest output"/);
  assert.match(styles, /\.log-latest-button\s*\{/);
});

test("the local training wiki opens as a searchable session tab", async () => {
  const [app, wiki, styles] = await Promise.all([
    fs.readFile("src/App.tsx", "utf8"),
    fs.readFile("src/TrainingWiki.tsx", "utf8"),
    fs.readFile("src/styles.css", "utf8"),
  ]);
  assert.match(app, /setWikiOpen\(true\)/);
  assert.match(app, /setWikiActive\(true\)/);
  assert.match(app, /className="session-tab-select"[\s\S]*?<b>Wiki<\/b>/);
  assert.match(app, /aria-label="Close Wiki"/);
  assert.match(app, /<TrainingWiki \/>/);
  assert.match(wiki, /aria-label="Search the training wiki"/);
  assert.match(wiki, /Wiki table of contents/);
  assert.match(wiki, /className="wiki-search-dock"/);
  assert.match(wiki, /Ai Training Wiki/);
  assert.doesNotMatch(wiki, /className="wiki-hero"/);
  assert.match(wiki, /aria-label="Wiki view controls"/);
  assert.match(wiki, /Hide contents/);
  assert.match(wiki, /Show contents/);
  assert.match(wiki, /Use continuous scrolling/);
  assert.match(wiki, /Use page mode/);
  assert.match(wiki, /Article \{activeIndex \+ 1\} of \{filtered\.length\}/);
  assert.match(wiki, /entryLinksRef\.current\[entryId\]\?\.scrollIntoView/);
  assert.match(wiki, /terms\.every\(\(term\) => haystack\.includes\(term\)\)/);
  assert.doesNotMatch(wiki, /https?:\/\//);
  assert.match(styles, /\.wiki-view\s*\{/);
  assert.match(styles, /\.wiki-search-dock\s*\{[^}]*position:\s*sticky/s);
  assert.match(styles, /\.wiki-search\s*\{/);
  assert.match(styles, /\.wiki-toc-scroll\s*\{[^}]*overflow-y:\s*auto/s);
  assert.match(styles, /\.wiki-tool-rail\s*\{/);
  assert.match(
    styles,
    /grid-template-columns:\s*42px minmax\(180px, 220px\) minmax\(0, 1fr\)/,
  );
  assert.match(styles, /\.wiki-note\s*\{[^}]*border-left:\s*1px/s);
});

test("the wiki documents every supported alignment method and its data", async () => {
  const wiki = await fs.readFile("src/TrainingWiki.tsx", "utf8");
  for (const method of [
    "Auto alignment",
    "DPO — Direct Preference Optimization",
    "IPO — Identity Preference Optimization",
    "SimPO — Simple Preference Optimization",
    "ORPO — Odds Ratio Preference Optimization",
    "CPO — Contrastive Preference Optimization",
    "KTO — Kahneman-Tversky Optimization",
    "PPO — Proximal Policy Optimization",
    "REINFORCE",
    "RLOO — REINFORCE Leave-One-Out",
    "GRPO — Group Relative Policy Optimization",
  ])
    assert.match(wiki, new RegExp(method));
  assert.match(wiki, /prompt.*chosen.*rejected/);
  assert.match(wiki, /prompt.*response.*reward/);
  assert.match(wiki, /Images, video, and audio/);
  assert.match(wiki, /Current alignment objectives operate on text/);
  assert.match(wiki, /“Online RL” does not mean online servers/);
});
