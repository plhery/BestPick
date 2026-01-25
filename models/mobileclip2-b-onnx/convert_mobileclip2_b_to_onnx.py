#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch
from huggingface_hub import hf_hub_download, list_repo_files


def _pick_weights_file(repo_id: str) -> str:
    files = list_repo_files(repo_id)
    pt_files = [f for f in files if f.endswith(".pt")]
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {repo_id}. Files: {files}")

    def score(name: str) -> int:
        n = name.lower()
        s = 0
        if "mobileclip2" in n:
            s += 2
        if "_b" in n or n.endswith("b.pt"):
            s += 2
        return s

    pt_files.sort(key=score, reverse=True)
    return pt_files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MobileCLIP2-B image/text encoders to ONNX.")
    parser.add_argument("--repo-id", default="apple/MobileCLIP2-B", help="Hugging Face repo id")
    parser.add_argument("--model-name", default="MobileCLIP2-B", help="OpenCLIP model name")
    parser.add_argument("--cache-dir", default="checkpoints", help="Directory to store downloaded weights")
    parser.add_argument("--out-dir", default="onnx", help="Output directory for ONNX files")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cache_dir = (root / args.cache_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure patched open_clip and vendor mobileclip are importable.
    open_clip_src = root / "open_clip" / "src"
    vendor_dir = root / "vendor"
    if not (open_clip_src / "open_clip").is_dir():
        raise RuntimeError("open_clip not found. Run ./setup_open_clip.sh first.")

    sys.path.insert(0, str(open_clip_src))
    sys.path.insert(0, str(vendor_dir))

    import open_clip  # noqa: E402
    from mobileclip.modules.common.mobileone import reparameterize_model  # noqa: E402

    weights_file = _pick_weights_file(args.repo_id)
    weights_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=weights_file,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )

    model_kwargs = {}
    if not (args.model_name.endswith("S3") or args.model_name.endswith("S4") or args.model_name.endswith("L-14")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

    model, _, _ = open_clip.create_model_and_transforms(
        args.model_name, pretrained=weights_path, **model_kwargs
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    model.eval()
    model = reparameterize_model(model)

    # Dummy inputs for export
    image = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    text = tokenizer(["a diagram"])  # shape [1, 77]

    image_encoder = getattr(model, "visual", None) or getattr(model, "image_encoder")
    text_encoder = getattr(model, "text", None) or getattr(model, "text_encoder")

    image_out = out_dir / "mobileclip2-b-image-encoder.onnx"
    text_out = out_dir / "mobileclip2-b-text-encoder.onnx"

    torch.onnx.export(
        image_encoder,
        image,
        str(image_out),
        input_names=["image"],
        output_names=["unnorm_image_features"],
        export_params=True,
        opset_version=args.opset,
        dynamic_axes={"image": {0: "batch"}, "unnorm_image_features": {0: "batch"}},
    )

    torch.onnx.export(
        text_encoder,
        text,
        str(text_out),
        input_names=["text"],
        output_names=["unnorm_text_features"],
        export_params=True,
        opset_version=args.opset,
        dynamic_axes={"text": {0: "batch"}, "unnorm_text_features": {0: "batch"}},
    )

    print(f"Saved: {image_out}")
    print(f"Saved: {text_out}")


if __name__ == "__main__":
    main()
