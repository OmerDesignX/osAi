#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Run this script on macOS 12 or newer." >&2
  exit 1
fi

node "$ROOT/releaseScripts/common/sync-version.mjs"
node "$ROOT/releaseScripts/common/cleanup-release.mjs"
cd "$ROOT"
node "$ROOT/releaseScripts/common/run-pnpm.mjs" install --frozen-lockfile
node "$ROOT/releaseScripts/common/run-pnpm.mjs" run release:build:macos
