#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOST_LABEL="${1:-Windows 10 or 11}"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ;;
  *)
    echo "Run the $HOST_LABEL release script in Git Bash on native 64-bit Windows." >&2
    exit 1
    ;;
esac

resolve_node() {
  local candidate=""
  if command -v node >/dev/null 2>&1; then
    command -v node
    return 0
  fi
  for candidate in \
    "/c/Program Files/nodejs/node.exe" \
    "$HOME/AppData/Local/Programs/nodejs/node.exe"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

resolve_pnpm() {
  local candidate=""
  if command -v pnpm >/dev/null 2>&1; then
    command -v pnpm
    return 0
  fi
  if command -v pnpm.cmd >/dev/null 2>&1; then
    command -v pnpm.cmd
    return 0
  fi
  for candidate in \
    "$HOME/AppData/Local/pnpm/pnpm.cmd" \
    "/c/Program Files/nodejs/pnpm.cmd"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! OSAI_NODE_BIN="$(resolve_node)"; then
  echo "Node.js 22 or newer was not found. Install Node.js, then reopen Git Bash." >&2
  exit 1
fi
if ! OSAI_PNPM_BIN="$(resolve_pnpm)"; then
  echo "pnpm 11 was not found. Install it, then reopen Git Bash." >&2
  exit 1
fi

export OSAI_NODE_BIN OSAI_PNPM_BIN
node() { "$OSAI_NODE_BIN" "$@"; }
pnpm() { "$OSAI_PNPM_BIN" "$@"; }
export -f node pnpm
export PATH="$(dirname "$OSAI_NODE_BIN"):$(dirname "$OSAI_PNPM_BIN"):$PATH"
export Path="$PATH"
export CI=true
export PNPM_DISABLE_SELF_UPDATE_CHECK=true
export NO_UPDATE_NOTIFIER=true
export npm_config_user_agent="pnpm/$(pnpm --version) node/$(node --version)"
export npm_execpath="$OSAI_PNPM_BIN"

NODE_MAJOR="$(node --version | sed -E 's/^v([0-9]+).*/\1/')"
if [[ ! "$NODE_MAJOR" =~ ^[0-9]+$ ]] || (( NODE_MAJOR < 22 )); then
  echo "Node.js 22 or newer is required; found $(node --version)." >&2
  exit 1
fi

echo "Using $(node --version) and pnpm $(pnpm --version)"
node "$ROOT/releaseScripts/common/cleanup-release.mjs"
bash "$ROOT/releaseScripts/common/prepare-source.sh"
cd "$ROOT"
export CSC_IDENTITY_AUTO_DISCOVERY=false
node "$ROOT/node_modules/electron-builder/cli.js" --win nsis --x64 --publish never
node scripts/verify-package.mjs windows x64 "$ROOT/release"
pnpm run release:stage:windows
node releaseScripts/common/cleanup-release.mjs

