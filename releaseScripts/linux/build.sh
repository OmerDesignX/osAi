#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this script on a current x64 Debian or Ubuntu host." >&2
  exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "The current Linux package supports x64 hosts only." >&2
  exit 1
fi

node "$ROOT/releaseScripts/common/cleanup-release.mjs"
bash "$ROOT/releaseScripts/common/prepare-source.sh"
cd "$ROOT"
pnpm exec electron-builder --linux deb --x64 --publish never
node scripts/verify-package.mjs linux x64 "$ROOT/release"
pnpm run release:stage:linux
node releaseScripts/common/cleanup-release.mjs

