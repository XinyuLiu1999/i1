#!/usr/bin/env python3
"""Download every model used by CVTG evaluation without loading the 11B VQA model."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvtg_evaluation import configure_cache


def download(cache):
    # Import only after setting the cache/endpoint environment variables.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel
    import clip
    import open_clip
    from paddleocr import PaddleOCR
    from t2v_metrics.models.vqascore_models.clip_t5_model import CLIP_T5_MODELS

    config = CLIP_T5_MODELS["clip-flant5-xxl"]
    print("Downloading CLIP-FlanT5-XXL weights (~23 GB)...", flush=True)
    snapshot = snapshot_download(config["model"]["path"], cache_dir=str(cache),
                                 allow_patterns=["*.json", "*.model", "pytorch_model*.bin"])
    model_config = json.loads((Path(snapshot) / "config.json").read_text())
    print("Downloading Flan-T5 tokenizer and CLIP vision tower...", flush=True)
    AutoTokenizer.from_pretrained(config["tokenizer"]["path"], use_fast=False, cache_dir=str(cache))
    vision = model_config["mm_vision_tower"]
    CLIPImageProcessor.from_pretrained(vision, cache_dir=str(cache))
    model = CLIPVisionModel.from_pretrained(vision, cache_dir=str(cache))
    del model
    gc.collect()
    print("Downloading official CLIP ViT-L/14...", flush=True)
    model, _ = clip.load("ViT-L/14", device="cpu", jit=False, download_root=str(cache / "clip"))
    del model
    gc.collect()
    print("Downloading OpenCLIP ViT-L-14 (OpenAI weights)...", flush=True)
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", cache_dir=str(cache))
    del model
    gc.collect()
    print("Downloading English PaddleOCR detection/recognition/angle models...", flush=True)
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False,
                    **{f"{kind}_model_dir": str(cache / "paddleocr" / kind) for kind in ("det", "rec", "cls")})
    del ocr
    head = Path(__file__).with_name("sa_0_4_vit_l_14_linear.pth")
    if not head.is_file():
        raise FileNotFoundError(f"Missing repository aesthetic head: {head}")
    print(f"All CVTG model assets are ready in {cache}; aesthetic head: {head}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", "--cache_dir", required=True)
    parser.add_argument("--use-hf-mirror", "--use_hf_mirror", action="store_true")
    parser.add_argument("--no-hf-mirror", "--no_hf_mirror", dest="use_hf_mirror", action="store_false")
    args = parser.parse_args()
    # Download/preprocess on CPU even on a multi-GPU host.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cache = configure_cache(args.cache_dir, args.use_hf_mirror)
    with (cache / ".cvtg-models.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        download(cache)


if __name__ == "__main__":
    main()
