import torch
import numpy as np
import cv2
import io, base64
import os
from PIL import Image as PILImage

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# ── Class definitions (must match COCO JSON training order) ──────────────────
CATEGORIES = ['shell', 'meat', 'residual biofouling', 'attached biofouling']

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../weights/model.pth")

# ── Detection thresholds ──────────────────────────────────────────────────────
# Global threshold — Detectron2 uses this for initial filtering
SCORE_THRESH = 0.1

# Per-class thresholds — applied after Detectron2 output
# Meat uses a stricter threshold to avoid false positives on empty shell halves
CLASS_THRESH = {
    "shell":               0.3,
    "residual biofouling": 0.3,
    "attached biofouling": 0.3,
    "meat":                0.6,   # stricter — empty shell nacre can look like meat
}

# Minimum pixel area for meat to be counted as real
# Detections smaller than this are treated as false positives
MIN_MEAT_PX = 500

# ── Overlay colors (RGBA) ─────────────────────────────────────────────────────
COLOR_MAP = {
    "shell":                (100, 200, 100, 120),
    "meat":                 (200, 120, 200, 140),
    "residual biofouling":  (220, 180,  60, 140),
    "attached biofouling":  (220,  80,  80, 160),
}

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    ))
    cfg.MODEL.ANCHOR_GENERATOR.SIZES         = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES          = 4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = SCORE_THRESH
    cfg.INPUT.MIN_SIZE_TEST                  = 800
    cfg.INPUT.MAX_SIZE_TEST                  = 1333
    cfg.MODEL.WEIGHTS                        = MODEL_PATH
    cfg.MODEL.DEVICE                         = "cpu"   # change to "cuda" if GPU available

    _predictor = DefaultPredictor(cfg)
    return _predictor


def _passes_filter(cls: str, score: float, mask_area: float) -> bool:
    """
    Returns True if this detection should be counted.
    Applies per-class score threshold and minimum meat area filter.
    """
    if score < CLASS_THRESH.get(cls, SCORE_THRESH):
        return False
    if cls == "meat" and mask_area < MIN_MEAT_PX:
        return False
    return True


def _blend(bgr_img: np.ndarray, overlay_rgba: np.ndarray) -> str:
    """
    Alpha-composites an RGBA overlay onto the original BGR image.
    Returns a base64-encoded PNG string for sending to the frontend.
    """
    rgb     = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    base    = PILImage.fromarray(rgb).convert("RGBA")
    top     = PILImage.fromarray(overlay_rgba, mode="RGBA")
    blended = PILImage.alpha_composite(base, top).convert("RGB")
    buf     = io.BytesIO()
    blended.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_inference(image_bgr: np.ndarray) -> dict:
    """
    Runs Mask R-CNN on a BGR numpy image.

    Returns
    -------
    dict with:
        class_area      — pixel counts per class (filtered)
        class_count     — instance counts per class (filtered)
        mean_score      — mean confidence per class (filtered)
        total_pixels    — total image pixel count
        overlay_b64     — full segmentation overlay (all classes) as base64 PNG
        shell_b64       — shell-only overlay as base64 PNG
        bio_b64         — biofouling-only overlay as base64 PNG
        meat_pixels_bgr — numpy array of meat pixel BGR values (for color analysis)
    """
    predictor = get_predictor()
    outputs   = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")

    pred_classes = instances.pred_classes.numpy()
    scores       = instances.scores.numpy()
    masks        = instances.pred_masks.numpy()

    # ── Accumulate filtered class stats ──────────────────────────────────────
    class_area  = {c: 0.0 for c in CATEGORIES}
    class_score = {c: []  for c in CATEGORIES}
    class_count = {c: 0   for c in CATEGORIES}
    meat_mask_all = np.zeros(image_bgr.shape[:2], dtype=bool)

    for i, cls_idx in enumerate(pred_classes):
        cls       = CATEGORIES[cls_idx]
        score     = float(scores[i])
        mask_area = float(masks[i].sum())

        if not _passes_filter(cls, score, mask_area):
            continue

        class_area[cls]  += mask_area
        class_score[cls].append(score)
        class_count[cls] += 1

        if cls == "meat":
            meat_mask_all |= masks[i].astype(bool)

    mean_score = {
        c: float(np.mean(v)) if v else 0.0
        for c, v in class_score.items()
    }

    h, w = image_bgr.shape[:2]

    # ── Full combined overlay (only filtered detections) ─────────────────────
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        combined = np.zeros((h, w), dtype=bool)
        for i, cls_idx in enumerate(pred_classes):
            if CATEGORIES[cls_idx] != cls:
                continue
            if not _passes_filter(cls, float(scores[i]), float(masks[i].sum())):
                continue
            combined |= masks[i].astype(bool)
        if combined.any():
            overlay[combined] = color
    overlay_b64 = _blend(image_bgr, overlay)

    # ── Shell-only overlay ────────────────────────────────────────────────────
    shell_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for i, cls_idx in enumerate(pred_classes):
        cls = CATEGORIES[cls_idx]
        if cls != "shell":
            continue
        if not _passes_filter(cls, float(scores[i]), float(masks[i].sum())):
            continue
        shell_overlay[masks[i].astype(bool)] = (100, 180, 255, 140)
    shell_b64 = _blend(image_bgr, shell_overlay)

    # ── Biofouling-only overlay ───────────────────────────────────────────────
    bio_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for i, cls_idx in enumerate(pred_classes):
        cls = CATEGORIES[cls_idx]
        if cls not in ("attached biofouling", "residual biofouling"):
            continue
        if not _passes_filter(cls, float(scores[i]), float(masks[i].sum())):
            continue
        bio_overlay[masks[i].astype(bool)] = (220, 80, 80, 160)
    bio_b64 = _blend(image_bgr, bio_overlay)

    # ── Meat pixels for color analysis ───────────────────────────────────────
    meat_pixels_bgr = image_bgr[meat_mask_all] if meat_mask_all.any() else None

    return {
        "class_area":      class_area,
        "class_count":     class_count,
        "mean_score":      mean_score,
        "total_pixels":    h * w,
        "overlay_b64":     overlay_b64,
        "shell_b64":       shell_b64,
        "bio_b64":         bio_b64,
        "meat_pixels_bgr": meat_pixels_bgr,
    }