#!/usr/bin/env python3
"""CVTG metrics, with bounded CLIP batches and strict inference failures.

The CLI accepts both legacy category/area folders and runner sample manifests.
See ../CVTG_EVALUATION.md for downloads and multi-GPU evaluation.
"""
from pathlib import Path
import re
import sys


class UnifiedMetricsEvaluator:
    def __init__(self, device="auto", cache_dir=None, use_hf_mirror=True):
        import torch
        import numpy as np
        self.torch, self.np = torch, np
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the evaluation environment")
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.models = {}
        self._load_models()

    def _load_models(self):
        import clip
        import open_clip
        import t2v_metrics
        from paddleocr import PaddleOCR
        cache = str(self.cache_dir) if self.cache_dir else None
        ocr_options = dict(use_angle_cls=True, lang="en", show_log=False,
                           use_gpu=self.device == "cuda")
        if self.cache_dir:
            for kind in ("det", "rec", "cls"):
                ocr_options[f"{kind}_model_dir"] = str(self.cache_dir / "paddleocr" / kind)
        self.models["ocr"] = PaddleOCR(**ocr_options)
        model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False,
                                      download_root=str(self.cache_dir / "clip") if self.cache_dir else None)
        self.models["clip"], self.models["clip_preprocess"] = model.eval(), preprocess
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", cache_dir=cache)
        self.models["openclip"] = model.to(self.device).eval()
        self.models["openclip_preprocess"] = preprocess
        head = self.torch.nn.Linear(768, 1)
        head.load_state_dict(self.torch.load(
            Path(__file__).with_name("sa_0_4_vit_l_14_linear.pth"), map_location="cpu"))
        self.models["aesthetic"] = head.to(self.device).eval()
        self.models["vqa"] = t2v_metrics.VQAScore(
            model="clip-flant5-xxl", device=self.device, **({"cache_dir": cache} if cache else {}))

    @staticmethod
    def extract_words_from_prompt(prompt):
        return [word for match in re.findall(r"'(.*?)'", prompt) for word in match.lower().split()]

    def compute_ocr_metrics(self, image_path, gt_words):
        import difflib
        import Levenshtein
        result = self.models["ocr"].ocr(str(image_path), cls=True)
        predicted = [word for page in (result or []) if page is not None
                     for line in page for word in line[1][0].lower().split()] or [""]
        distances = []
        for word in gt_words:
            best = difflib.get_close_matches(word, predicted, n=1, cutoff=0)
            distances.append(1 - Levenshtein.distance(word, best[0]) /
                             (max(len(word), len(best[0])) + 1e-5) if best else 0.0)
        return len(gt_words), sum(word in predicted for word in gt_words), distances

    def compute_clip_score_batch(self, image_paths, texts):
        import clip
        from PIL import Image
        from packaging import version
        from sklearn.preprocessing import normalize
        torch, np = self.torch, self.np
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(self.models["clip_preprocess"](image))
        image_inputs = torch.stack(images).to(self.device)
        text_inputs = clip.tokenize(["A photo depicts " + text for text in texts], truncate=True).to(self.device)
        with torch.no_grad():
            image_features = self.models["clip"].encode_image(image_inputs).cpu().numpy()
            text_features = self.models["clip"].encode_text(text_inputs).cpu().numpy()
        # Preserve the original CLIPScore normalization and 2.5 scale.
        if version.parse(np.__version__) < version.parse("1.21"):
            image_features = normalize(image_features, axis=1)
            text_features = normalize(text_features, axis=1)
        else:
            image_features = image_features / np.sqrt(np.sum(image_features**2, axis=1, keepdims=True))
            text_features = text_features / np.sqrt(np.sum(text_features**2, axis=1, keepdims=True))
        return (2.5 * np.clip(np.sum(image_features * text_features, axis=1), 0, None)).tolist()

    def compute_vqa_score(self, image_path, text):
        with self.torch.no_grad():
            return float(self.models["vqa"](images=[str(image_path)], texts=[text]).cpu().numpy().mean())

    def compute_aesthetic_score(self, image_path):
        from PIL import Image
        with Image.open(image_path) as image:
            inputs = self.models["openclip_preprocess"](image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            features = self.models["openclip"].encode_image(inputs)
            features /= features.norm(dim=-1, keepdim=True)
            return float(self.models["aesthetic"](features).cpu().numpy().item())

    def score_samples(self, samples, batch_size=16):
        if batch_size < 1:
            raise ValueError("CLIP batch size must be positive")
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            scores = self.compute_clip_score_batch([r["image"] for r in batch], [r["prompt"] for r in batch])
            if len(scores) != len(batch):
                raise RuntimeError("CLIP returned an incomplete batch")
            for sample, clip_score in zip(batch, scores):
                total, correct, ned = self.compute_ocr_metrics(
                    sample["image"], self.extract_words_from_prompt(sample["prompt"]))
                yield {**sample, "total_words": total, "correct_words": correct, "ned_word_data": ned,
                       "clipscore": float(clip_score),
                       "vqascore": self.compute_vqa_score(sample["image"], sample["prompt"]),
                       "aesthetic_score": self.compute_aesthetic_score(sample["image"])}


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cvtg_evaluation import main as evaluate
    evaluate()


if __name__ == "__main__":
    main()
