#!/bin/sh
# Build the browser version into web/dist: wasm, JS bindings and the page.
# Needs the wasm32-unknown-unknown target and wasm-bindgen-cli matching Cargo.lock's
# wasm-bindgen; uses wasm-opt (binaryen) when installed.
set -eu
cd "$(dirname "$0")/.."
cargo build --profile web --target wasm32-unknown-unknown -p wetpaint-web
BINDGEN=${WASM_BINDGEN:-$(command -v wasm-bindgen || echo "$HOME/.cargo/bin/wasm-bindgen")}
mkdir -p web/dist
"$BINDGEN" --target web --no-typescript --out-dir web/dist --out-name wetpaint \
  target/wasm32-unknown-unknown/web/wetpaint_web.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -Os --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
    --enable-mutable-globals --enable-reference-types \
    web/dist/wetpaint_bg.wasm -o web/dist/wetpaint_bg.wasm
fi
cp web/index.html web/dist/index.html
# the claude.ai artifact page: the same app with the JS bindings inline (the artifact frame
# only loads scripts from the page itself or a few CDNs), and the module gzipped in 1 MiB
# parts (wetpaint-<build>-gz-N.wasm, named .wasm to be served) that the page joins and
# unpacks, small enough to upload. <build> is a hash of the module, so a new build's parts
# can be uploaded beside the old ones before the page switches to them.
python3 -I - <<'PY'
import glob, gzip, hashlib, os
wasm = open("web/dist/wetpaint_bg.wasm", "rb").read()
build = hashlib.sha256(wasm).hexdigest()[:8]
page = open("web/artifact.html").read()
js = open("web/dist/wetpaint.js").read()
assert "/*WETPAINT_JS*/" in page and "/*WETPAINT_BUILD*/" in page
page = page.replace("/*WETPAINT_BUILD*/", build).replace("/*WETPAINT_JS*/", js)
open("web/dist/artifact.html", "w").write(page)
for stale in glob.glob("web/dist/wetpaint-*gz*.wasm") + glob.glob("web/dist/wetpaint.wasm.gz.*"):
    os.remove(stale)
gz = gzip.compress(wasm, 9)
part = 1 << 20
for i in range(0, len(gz), part):
    open(f"web/dist/wetpaint-{build}-gz-{i // part}.wasm", "wb").write(gz[i:i + part])
PY
ls -l web/dist
