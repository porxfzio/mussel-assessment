import torch
import numpy as np
import cv2
import io, base64
import os
from PIL import Image as PILImage

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# Exact order from your COCO JSON (sorted by category ID)
CATEGORIES = ['shell', 'meat', 'residual biofouling', 'attached biofouling']

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "../weights/model.pth")
SCORE_THRESH = 0.1

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
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

def run_inference(image_bgr: np.ndarray) -> dict:
    predictor = get_predictor()
    outputs   = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")

    pred_classes = instances.pred_classes.numpy()
    scores       = instances.scores.numpy()
    masks        = instances.pred_masks.numpy()

    class_area  = {c: 0.0 for c in CATEGORIES}
    class_score = {c: [] for c in CATEGORIES}
    class_count = {c: 0  for c in CATEGORIES}
    meat_mask_all = np.zeros(image_bgr.shape[:2], dtype=bool)

    for i, cls_idx in enumerate(pred_classes):
        cls = CATEGORIES[cls_idx]
        class_area[cls]  += float(masks[i].sum())
        class_score[cls].append(float(scores[i]))
        class_count[cls] += 1
        if cls == 'meat':
            meat_mask_all |= masks[i].astype(bool)

    mean_score = {c: float(np.mean(v)) if v else 0.0 for c, v in class_score.items()}

    h, w = image_bgr.shape[:2]

    # ── blend helper defined FIRST ──────────────────────────────
    def blend(bgr_img, overlay_rgba):
        rgb  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        base = PILImage.fromarray(rgb).convert("RGBA")
        top  = PILImage.fromarray(overlay_rgba, mode="RGBA")
        blended = PILImage.alpha_composite(base, top).convert("RGB")
        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # Combined overlay blended onto original
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    color_map = {
        "shell":                (100, 200, 100, 120),
        "meat":                 (200, 120, 200, 140),
        "residual biofouling":  (220, 180,  60, 140),
        "attached biofouling":  (220,  80,  80, 160),
    }
    for cls, color in color_map.items():
        combined = np.zeros((h, w), dtype=bool)
        for i, cls_idx in enumerate(pred_classes):
            if CATEGORIES[cls_idx] == cls:
                combined |= masks[i].astype(bool)
        if combined.any():
            overlay[combined] = color
    overlay_b64 = blend(image_bgr, overlay)

    # Shell mask blended onto original
    shell_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for i, cls_idx in enumerate(pred_classes):
        if CATEGORIES[cls_idx] == "shell":
            shell_overlay[masks[i].astype(bool)] = (100, 180, 255, 140)
    shell_b64 = blend(image_bgr, shell_overlay)

    # Biofouling blended onto original
    bio_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for i, cls_idx in enumerate(pred_classes):
        if CATEGORIES[cls_idx] in ("attached biofouling", "residual biofouling"):
            bio_overlay[masks[i].astype(bool)] = (220, 80, 80, 160)
    bio_b64 = blend(image_bgr, bio_overlay)

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