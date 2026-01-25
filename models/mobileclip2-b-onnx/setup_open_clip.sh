#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

if [ ! -d "$ROOT_DIR/open_clip" ]; then
  git clone https://github.com/mlfoundations/open_clip.git "$ROOT_DIR/open_clip"
fi

if [ ! -f "$ROOT_DIR/open_clip/src/open_clip/mobileclip2.py" ]; then
  cp -R "$ROOT_DIR/vendor/mobileclip2"/* "$ROOT_DIR/open_clip/src/open_clip/"
  (cd "$ROOT_DIR/open_clip" && git apply "$ROOT_DIR/vendor/mobileclip2/open_clip_inference_only.patch")
fi

echo "open_clip ready in $ROOT_DIR/open_clip"
