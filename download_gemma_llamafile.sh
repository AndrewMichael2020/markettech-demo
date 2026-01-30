#!/usr/bin/env bash
set -euo pipefail

# Download Mozilla Gemma 3 4B IT llamafile into the repo.
# Usage: ./download_gemma_llamafile.sh

URL="https://huggingface.co/Mozilla/gemma-3-4b-it-llamafile/resolve/main/google_gemma-3-4b-it-Q6_K.llamafile?download=true"
OUT="google_gemma-3-4b-it-Q6_K.llamafile"

if [[ -f "$OUT" ]]; then
  echo "[info] $OUT already exists; skipping download."
  exit 0
fi

echo "[info] Downloading Gemma llamafile (~several GB, may take a while)..."
curl -L "$URL" -o "$OUT"
chmod +x "$OUT"

echo "[info] Download complete: $OUT (executable)."
