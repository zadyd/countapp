#!/bin/sh
# 下载网页版运行所需的推理引擎（约 20MB，未纳入仓库以控制体积）
# 用法：在 web/ 目录下执行 sh fetch-deps.sh
set -e
VER=1.18.0
BASE="https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-$VER.tgz"
TMP=$(mktemp -d)
echo "下载 onnxruntime-web $VER ..."
curl -sL "$BASE" -o "$TMP/ort.tgz"
tar xzf "$TMP/ort.tgz" -C "$TMP"
cp "$TMP/package/dist/ort.min.js" .
cp "$TMP/package/dist/ort-wasm-simd.wasm" .
cp "$TMP/package/dist/ort-wasm-simd-threaded.wasm" .
rm -rf "$TMP"
echo "完成：ort.min.js / ort-wasm-simd.wasm / ort-wasm-simd-threaded.wasm"
