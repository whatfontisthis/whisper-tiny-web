#!/bin/bash
set -e
cd "$(dirname "$0")"

BASE="https://huggingface.co"
FILES=(
  "config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "preprocessor_config.json"
  "generation_config.json"
  "onnx/encoder_model_quantized.onnx"
  "onnx/decoder_model_merged_quantized.onnx"
)

download_model() {
  local repo="$1"
  local dest="models/$repo"
  echo "=== $repo ==="
  mkdir -p "$dest/onnx"
  for f in "${FILES[@]}"; do
    local url="$BASE/$repo/resolve/main/$f"
    local out="$dest/$f"
    if [ -f "$out" ]; then
      echo "skip $f (exists)"
    else
      echo "get $f"
      curl -L --fail -o "$out" "$url"
    fi
  done
}

download_model "Xenova/whisper-tiny.en"
download_model "Xenova/whisper-small.en"
download_model "Xenova/whisper-small"

echo "Done. Total:"
du -sh models/
