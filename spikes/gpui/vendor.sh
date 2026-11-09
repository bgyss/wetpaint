#!/bin/sh
# Fetch gpui (Zed's tree at the pinned revision) into ../gpui-vendor/zed and apply
# patches/external-texture.patch. The tree sits beside this directory, not inside it:
# Cargo treats path dependencies under a workspace's directory as its members, and
# gpui's crates need Zed's own workspace manifest.
set -eu
cd "$(dirname "$0")"
REV=c7bea343838f42b4c14151393e41f0bbe033c95e
DIR=../gpui-vendor/zed
if [ -d "$DIR" ]; then
  echo "$DIR exists; remove it to fetch again" >&2
  exit 1
fi
mkdir -p ../gpui-vendor
git clone --filter=blob:none --no-checkout https://github.com/zed-industries/zed.git "$DIR"
git -C "$DIR" sparse-checkout set --no-cone '/crates/' '/Cargo.toml' '/tooling/' '/assets/fonts/' '/LICENSE*'
git -C "$DIR" checkout -q "$REV"
git -C "$DIR" apply "$PWD/patches/external-texture.patch"
git -C "$DIR" status --short
