#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"
mojo build --emit shared-lib -I "$repo_dir/src" \
    "$repo_dir/src/capi.mojo" \
    -o "$repo_dir/dist/libmojo-skyfield.so"
# `python/mojo_skyfield/_lib.py` loads `build/capi.so` and rebuilds it when it
# is older than `src/`, so keep that copy fresh instead of paying a second
# build.
cp "$repo_dir/dist/libmojo-skyfield.so" "$repo_dir/build/capi.so"
