#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "The macOS icon must be generated on macOS." >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ICONSET="$WORK/icon-macos.iconset"
mkdir -p "$ICONSET"

SOURCE_HASH="$(shasum -a 256 "$ROOT/build/icon.png" | awk '{print $1}')"
HASH_FILE="$ROOT/build/icon-macos.source.sha256"
ICNS_FILE="$ROOT/build/icon-macos.icns"

if [[ -f "$ICNS_FILE" && -f "$HASH_FILE" ]] && \
   [[ "$(tr -d '[:space:]' < "$HASH_FILE")" == "$SOURCE_HASH" ]]; then
  iconutil -c iconset "$ICNS_FILE" -o "$WORK/verified.iconset"
  COUNT="$(find "$WORK/verified.iconset" -type f -name '*.png' | wc -l | tr -d ' ')"
  if [[ "$COUNT" -ge 10 ]]; then
    echo "Verified the cached macOS icon for the current source artwork."
    exit 0
  fi
fi

xcrun swift "$SCRIPT_DIR/render-icon.swift" \
  "$ROOT/build/icon.png" "$WORK/icon-1024.png" 0.88

sips -z 16 16 "$WORK/icon-1024.png" --out "$ICONSET/icon_16x16.png" >/dev/null
sips -z 32 32 "$WORK/icon-1024.png" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$WORK/icon-1024.png" --out "$ICONSET/icon_32x32.png" >/dev/null
sips -z 64 64 "$WORK/icon-1024.png" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$WORK/icon-1024.png" --out "$ICONSET/icon_128x128.png" >/dev/null
sips -z 256 256 "$WORK/icon-1024.png" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$WORK/icon-1024.png" --out "$ICONSET/icon_256x256.png" >/dev/null
sips -z 512 512 "$WORK/icon-1024.png" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$WORK/icon-1024.png" --out "$ICONSET/icon_512x512.png" >/dev/null
cp "$WORK/icon-1024.png" "$ICONSET/icon_512x512@2x.png"

iconutil -c icns "$ICONSET" -o "$ICNS_FILE"
iconutil -c iconset "$ICNS_FILE" -o "$WORK/verified.iconset"
COUNT="$(find "$WORK/verified.iconset" -type f -name '*.png' | wc -l | tr -d ' ')"
if [[ "$COUNT" -lt 10 ]]; then
  echo "The generated macOS icon is missing sizes." >&2
  exit 1
fi

FINAL_SOURCE_HASH="$(shasum -a 256 "$ROOT/build/icon.png" | awk '{print $1}')"
if [[ "$FINAL_SOURCE_HASH" != "$SOURCE_HASH" ]]; then
  echo "build/icon.png changed while preparing the macOS icon." >&2
  exit 1
fi

printf '%s\n' "$SOURCE_HASH" > "$HASH_FILE"

echo "Generated build/icon-macos.icns with osCode-style rounded artwork and padding."
