"""CPU-only OCR normalization and height-weighted masks in the VAE latent grid."""
from collections import Counter
import json
import math

import numpy as np
from PIL import Image, ImageDraw

MASK_VERSION = "ocr-height-weighted-max-v2"
LATENT_FACTOR = 8


def ocr_items(raw):
    if raw is None or raw == "":
        return []
    value = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(value, dict):
        value = value.get("res", value)
        if not isinstance(value, dict):
            raise ValueError("OCR res must be a dictionary")
        texts = value.get("rec_texts", [])
        scores = value.get("rec_scores", [])
        polygons = value.get("rec_polys", [])
        boxes = value.get("rec_boxes", [])
        if any(not isinstance(field, list) for field in (texts, scores, polygons, boxes)):
            raise ValueError("OCR text/score/geometry fields must be lists")
        value = [dict(text=text, score=scores[i] if i < len(scores) else None,
                      poly=polygons[i] if i < len(polygons) else None,
                      box=boxes[i] if i < len(boxes) else None)
                 for i, text in enumerate(texts)]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("OCR must contain region dictionaries")
    return value


def polygon(item):
    points = item.get("poly")
    if points is None or points == []:
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("No polygon or xyxy box")
        x0, y0, x1, y1 = map(float, box)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Empty/reversed box")
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Invalid polygon")
    # Reject degenerate polygons rather than inventing a rectangle.
    area = abs(np.dot(points[:, 0], np.roll(points[:, 1], 1))
               - np.dot(points[:, 1], np.roll(points[:, 0], 1))) / 2
    if area <= 0:
        raise ValueError("Zero-area polygon")
    return points


def make_mask(raw, width, height, bucket, min_confidence=0.8, min_height=4.0,
              reference_height=4.0):
    """Max-combine per-region height weights, then average each 8x8 cell.

    Heights are measured after the exact fit-and-pad resize, in training pixels.
    Each accepted region receives ``min(1, reference_height / region_height)``.
    Drawing lower weights first makes overlaps take the maximum region weight,
    which is deterministic and is the weighted analogue of a binary union.
    """
    bh, bw = bucket
    if min(width, height, bh, bw) <= 0 or bh % LATENT_FACTOR or bw % LATENT_FACTOR:
        raise ValueError("Invalid source/bucket dimensions")
    if (not 0 <= min_confidence <= 1 or not math.isfinite(min_height) or min_height < 0
            or not math.isfinite(reference_height) or reference_height <= 0):
        raise ValueError("Invalid OCR thresholds")
    stats = Counter()
    regions = []
    scale = min(bw / width, bh / height)
    nw, nh = max(1, min(bw, round(width * scale))), max(1, min(bh, round(height * scale)))
    sx, sy, dx, dy = nw / width, nh / height, (bw - nw) // 2, (bh - nh) // 2
    try:
        items = ocr_items(raw)
    except (TypeError, ValueError, KeyError):
        items = []
        stats["malformed_ocr"] += 1
    for item in items:
        stats["regions_seen"] += 1
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            stats["empty_text"] += 1
            continue
        try:
            confidence = float(item.get("score"))
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid confidence")
        except (ValueError, TypeError):
            stats["invalid_confidence"] += 1
            continue
        if confidence < min_confidence:
            stats["low_confidence"] += 1
            continue
        try:
            points = polygon(item)
        except (TypeError, ValueError, OverflowError):
            stats["invalid_geometry"] += 1
            continue
        if ((points < 0).any() or (points[:, 0] > width).any()
                or (points[:, 1] > height).any()):
            stats["out_of_bounds"] += 1
            continue
        points = points * [sx, sy] + [dx, dy]
        region_height = float(np.ptp(points[:, 1]))
        if region_height < min_height:
            stats["below_min_height"] += 1
            continue
        # ImageDraw uses inclusive integer pixel coordinates. Clip to the actual
        # resized content so an edge polygon never marks letterbox padding.
        points[:, 0] = np.clip(points[:, 0], dx, dx + nw - 1)
        points[:, 1] = np.clip(points[:, 1], dy, dy + nh - 1)
        weight = min(1.0, reference_height / region_height)
        regions.append((weight, points))
        stats["regions_used"] += 1
        stats["regions_downweighted"] += weight < 1.0
        stats["region_height_sum"] += region_height
        stats["region_weight_sum"] += weight
    canvas = Image.new("F", (bw, bh), 0.0)
    draw = ImageDraw.Draw(canvas)
    for weight, points in sorted(regions, key=lambda value: value[0]):
        draw.polygon([tuple(p) for p in points], fill=weight)
    pixels = np.asarray(canvas, dtype=np.float32)
    if (pixels < 0).any() or (pixels > 1).any() or not np.isfinite(pixels).all():
        raise ValueError("Invalid rasterized OCR weights")
    latent_weights = pixels.reshape(bh // 8, 8, bw // 8, 8).mean(axis=(1, 3))
    return latent_weights.astype("<f2"), dict(stats)
