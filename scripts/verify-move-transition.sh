#!/usr/bin/env bash
# Isolated Move Transition scene-motion and HDR pixel-preservation regression check.
set -euo pipefail
if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 <move-transition.so> <plugin-data-dir> [libmediapipe.so]" >&2
  exit 2
fi
repo="$(cd "$(dirname "$0")/.." && pwd)"
module="$(realpath "$1")"
plugin_data="$(realpath "$2")"
tmp="$(mktemp -d /tmp/verify-move-transition-XXXXXX)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"
cp "$module" "$tmp/bin/move-transition.so"
if (( $# == 3 )); then
  cp "$(realpath "$3")" "$tmp/bin/libmediapipe.so"
fi
g++ -std=c++17 -O0 -g -I/usr/include/obs \
  "$repo/scripts/verify-move-transition.cpp" \
  -o "$tmp/obs-move-probe" $(pkg-config --libs libobs)
for mode in sdr hdr; do
  mkdir -p "$tmp/home-$mode"
  log="$tmp/$mode.log"
  if ! HOME="$tmp/home-$mode" XDG_CONFIG_HOME="$tmp/home-$mode" \
    xvfb-run -a timeout 35 "$tmp/obs-move-probe" \
    "$tmp/bin/move-transition.so" "$plugin_data" "$mode" >"$log" 2>&1; then
    echo "MOVE_TRANSITION $mode failed:" >&2
    tail -60 "$log" >&2
    exit 1
  fi
  if ! grep -q '^MOVE_VERIFY_PASS$' "$log"; then
    echo "MOVE_TRANSITION $mode exited without test completion" >&2
    tail -50 "$log" >&2
    exit 1
  fi
  if grep -qE 'Double destroy just occurred|source\(s\) were remaining' "$log"; then
    echo "MOVE_TRANSITION $mode had an OBS source-lifetime error" >&2
    tail -50 "$log" >&2
    exit 1
  fi
  grep -E '^COLOR_SPACE|^CLOCK|^RENDER|^HDR_PIXEL|^SCENE_MOVED|^MOVE_VERIFY_PASS' "$log"
  echo "MOVE_TRANSITION $mode PASS=1"
done
