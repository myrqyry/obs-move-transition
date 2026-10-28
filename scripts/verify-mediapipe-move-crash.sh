#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <move-transition.so> <libmediapipe.so> <plugin-data-dir> <256x256-RGBA-fixture>" >&2
  exit 2
fi
module="$(realpath "$1")"
lib="$(realpath "$2")"
data="$(realpath "$3")"
face="$(realpath "$4")"
repo="$(cd "$(dirname "$0")/.." && pwd)"
dir="$(mktemp -d /tmp/verify-mp-move-XXXXXX)"
trap 'rm -rf "$dir"' EXIT
mkdir -p "$dir/bin"
cp "$module" "$dir/bin/move-transition.so"
cp "$lib" "$dir/bin/libmediapipe.so"
g++ -std=c++17 -O0 -g -I/usr/include/obs "$repo/scripts/verify-mediapipe-move-crash.cpp" \
  -o "$dir/obs-probe" $(pkg-config --libs libobs)
export MESA_SHADER_CACHE_DISABLE=true
for mode in default face body both; do
  mkdir -p "$dir/home-$mode"
  if ! HOME="$dir/home-$mode" XDG_CONFIG_HOME="$dir/home-$mode" \
      xvfb-run -a timeout 35 "$dir/obs-probe" "$dir/bin/move-transition.so" "$data" \
      "$mode" render "$face" >"$dir/$mode.log" 2>&1; then
    echo "MEDIAPIPE_MOVE $mode failed:"
    tail -45 "$dir/$mode.log"
    exit 1
  fi
  if ! grep -q 'REPRO_PASS' "$dir/$mode.log"; then
    echo "MEDIAPIPE_MOVE $mode exited without test completion"
    tail -30 "$dir/$mode.log"
    exit 1
  fi
  if [[ "$mode" == face || "$mode" == both || "$mode" == default ]]; then
    grep -q 'Face tracking acquired (478 landmarks)' "$dir/$mode.log" || {
      echo "MEDIAPIPE_MOVE $mode did not detect the test face"; tail -30 "$dir/$mode.log"; exit 1;
    }
  fi
  if [[ "$mode" == body || "$mode" == both ]]; then
    grep -q 'Pose tracking acquired (33 landmarks)' "$dir/$mode.log" || {
      echo "MEDIAPIPE_MOVE $mode did not detect the test pose"; tail -30 "$dir/$mode.log"; exit 1;
    }
  fi
  if [[ "$mode" == face || "$mode" == both || "$mode" == default ]]; then
    grep -q 'Face tracking lost (0 landmarks)' "$dir/$mode.log" || {
      echo "MEDIAPIPE_MOVE $mode did not clear the face when input disappeared"; tail -30 "$dir/$mode.log"; exit 1;
    }
  fi
  if [[ "$mode" == body || "$mode" == both ]]; then
    grep -q 'Pose tracking lost (0 landmarks)' "$dir/$mode.log" || {
      echo "MEDIAPIPE_MOVE $mode did not clear the pose when input disappeared"; tail -30 "$dir/$mode.log"; exit 1;
    }
  fi
  echo "MEDIAPIPE_MOVE $mode acquired=1 cleared=1 crash=0 PASS=1"
done
