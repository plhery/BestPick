# MobileCLIP2-B ONNX export

This folder downloads **MobileCLIP2-B** weights and exports the image and text encoders to ONNX so they can be wired into transformers.js later.

## Setup

```bash
./setup_open_clip.sh

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e open_clip
```

## Export

```bash
source .venv/bin/activate
python convert_mobileclip2_b_to_onnx.py
```

Outputs (exported with opset 18):

- `onnx/mobileclip2-b-image-encoder.onnx`
- `onnx/mobileclip2-b-text-encoder.onnx`

## Notes for transformers.js later

- Outputs are **unnormalized** embeddings; L2-normalize before computing cosine similarities.
- Text input is token IDs shaped `[batch, 77]` (CLIP BPE vocab size 49408).
- Image input is float32 tensor `[batch, 3, 224, 224]`.
- For MobileCLIP2-B, Apple recommends `image_mean=(0,0,0)` and `image_std=(1,1,1)`.

## Sources

- Apple MobileCLIP2 instructions for OpenCLIP patching and checkpoint download.
- AXERA-TECH repo shows a simple ONNX export pattern for MobileCLIP2.
