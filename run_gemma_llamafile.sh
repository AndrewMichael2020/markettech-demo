#!/usr/bin/env bash
set -euo pipefail

# Run the Gemma llamafile with an OpenAI-compatible HTTP server.
# Usage: ./run_gemma_llamafile.sh

MODEL_FILE="google_gemma-3-4b-it-Q6_K.llamafile"
PORT="8080"
HOST="0.0.0.0"

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "[error] $MODEL_FILE not found. Run ./download_gemma_llamafile.sh first."
  exit 1
fi

# You can change THREADS or other options if desired.
THREADS="4"

echo "[info] Starting Gemma llamafile server on http://$HOST:$PORT ..."

echo "[hint] In another shell, you can set:"
echo "       export GEMMA_API_BASE=http://127.0.0.1:$PORT/v1"

eval "./$MODEL_FILE --server --host $HOST --port $PORT --threads $THREADS"
