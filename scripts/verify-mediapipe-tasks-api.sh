#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <libmediapipe.so> <face_landmarker.task> <pose_landmarker.task> <256x256-RGBA-fixture>" >&2
  exit 2
fi
repo="$(cd "$(dirname "$0")/.." && pwd)"
dir="$(mktemp -d /tmp/verify-mp-tasks-XXXXXX)"
trap 'rm -rf "$dir"' EXIT
g++ -std=c++17 -O2 -I"$repo" -I/usr/include/obs \
  "$repo/scripts/verify-mediapipe-tasks-api.cpp" -o "$dir/probe" -ldl
timeout 35 "$dir/probe" "$1" "$2" "$3" "$4"
