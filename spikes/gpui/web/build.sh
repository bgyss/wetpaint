#!/bin/sh
# Build the spike for the browser into web/dist. Needs the wasm32-unknown-unknown target
# for the toolchain in rust-toolchain.toml and wasm-bindgen-cli matching Cargo.lock's
# wasm-bindgen (WASM_BINDGEN=/path/to/wasm-bindgen); uses wasm-opt when installed.
set -eu
cd "$(dirname "$0")/.."
CARGO_PROFILE_RELEASE_DEBUG=false cargo build --release --target wasm32-unknown-unknown
BINDGEN=${WASM_BINDGEN:-wasm-bindgen}
mkdir -p web/dist
"$BINDGEN" --target web --no-typescript --out-dir web/dist --out-name wetpaint_gpui \
  target/wasm32-unknown-unknown/release/wetpaint-gpui.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -O2 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
    --enable-mutable-globals --enable-reference-types \
    web/dist/wetpaint_gpui_bg.wasm -o web/dist/wetpaint_gpui_bg.wasm
fi
cp web/index.html web/dist/index.html
ls -l web/dist
