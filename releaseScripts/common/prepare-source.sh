#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

node "$SCRIPT_DIR/sync-version.mjs"
cd "$ROOT"
CI=true \
  PNPM_DISABLE_SELF_UPDATE_CHECK=true \
  NO_UPDATE_NOTIFIER=true \
  pnpm install --frozen-lockfile --prefer-offline
pnpm run release:check-disk
pnpm run format:check
pnpm test
NODE_OPTIONS=--max-old-space-size=4096 pnpm exec vite build

