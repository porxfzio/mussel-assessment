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

# ── Global score threshold (Detectron2 initial filter) ───────────────────────
SCORE_THRESH = 0.1

# ── Per-class score thresholds ────────────────────────────────────────────────
CLASS_THRESH = {
    "shell":               0.3,
    "residual biofouling": 0.3,
    "attached biofouling": 0.3,
    "meat":                0.5,
}

# ── Minimum meat pixel area — separate per photo type ────────────────────────
# meat.jpg  : high threshold filters nacre false positives from the empty
#             shell half in butterfly-style opened mussel photos.
# side photos: low threshold preserves small exposed meat regions on
#              broken shells, which are a genuine Grade C signal.
MIN_MEAT_PX_MEAT_PHOTO  = 500   # meat.jpg — strict, avoids nacre false positives
MIN_MEAT_PX_SHELL_PHOTO = 50    # side A/B — lenient, allows broken shell meat

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


def _keep_largest_meat_only(pred_classes, scores, masks, image_bgr,
                             min_meat_px: int) -> set:
    """
    Among all meat detections, keep only the one that:
    1. Has warm meat-like color (orange/cream) — primary filter
    2. Is the largest among those that pass color check
    If NO detection passes color check, fall back to largest overall.

    min_meat_px is passed in from run_inference so that shell photos
    and meat photos can use different minimum area thresholds.
    """
    meat_candidates = []
    for i, cls_idx in enumerate(pred_classes):
        if CATEGORIES[cls_idx] != "meat":
            continue
        score     = float(scores[i])
        mask_area = float(masks[i].sum())
        if score >= CLASS_THRESH["meat"] and mask_area >= min_meat_px:
            meat_candidates.append((i, mask_area, score))

    if len(meat_candidates) == 0:
        return set()
    if len(meat_candidates) == 1:
        return {meat_candidates[0][0]}

    # ── Score each candidate by color ────────────────────────────────────────
    def _meat_color_score(mask, image_bgr):
        """
        Returns a score 0-3 based on how much the region looks like real meat.
        Higher = more meat-like color.
        3 = warm orange/cream (real meat)
        2 = ambiguous warm tone
        1 = ambiguous cool tone
        0 = nacre / bluish / desaturated (false positive)
        """
        pixels_bgr = image_bgr[mask.astype(bool)]
        if len(pixels_bgr) == 0:
            return 0

        pixels_hsv = cv2.cvtColor(
            pixels_bgr.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2HSV
        ).reshape(-1, 3).astype(float)

        mean_hue = pixels_hsv[:, 0].mean()   # 0-180
        mean_sat = pixels_hsv[:, 1].mean()   # 0-255
        mean_val = pixels_hsv[:, 2].mean()   # 0-255

        print(f"[MEAT CANDIDATE] hue={mean_hue:.1f} sat={mean_sat:.1f} val={mean_val:.1f}", end=" → ")

        # Nacre: bluish/cyan hue OR very low saturation
        if (60 <= mean_hue <= 140) or mean_sat < 35:
            print("NACRE (score 0)")
            return 0

        # Real meat: warm orange/cream
        if (mean_hue <= 30 or mean_hue >= 150) and mean_sat >= 50:
            print("REAL MEAT (score 3)")
            return 3

        # Warm but less saturated (cream/pale meat)
        if mean_hue <= 40 and mean_sat >= 35:
            print("PALE MEAT (score 2)")
            return 2

        # Ambiguous
        print("AMBIGUOUS (score 1)")
        return 1

    # Score each candidate
    scored = []
    for i, mask_area, score in meat_candidates:
        color_score = _meat_color_score(masks[i], image_bgr)
        scored.append((i, mask_area, score, color_score))
        print(f"  area={int(mask_area)}px conf={score:.2f} color_score={color_score}")

    # Sort by color score first (higher = better), then by area (larger = better)
    scored.sort(key=lambda x: (x[3], x[1]), reverse=True)
    best = scored[0]
    kept_idx = best[0]

    # Log discarded ones
    for i, area, conf, cscore in scored[1:]:
        print(f"[MEAT FILTER] Discarded — area={int(area)}px conf={conf:.2f} color_score={cscore}")
    print(f"[MEAT FILTER] Kept — area={int(best[1])}px conf={best[2]:.2f} color_score={best[3]}")

    return {kept_idx}


def _passes_filter(cls: str, score: float, mask_area: float,
                   min_meat_px: int) -> bool:
    """Base filter: per-class score threshold + minimum meat pixel area."""
    if score < CLASS_THRESH.get(cls, SCORE_THRESH):
        return False
    if cls == "meat" and mask_area < min_meat_px:
        return False
    return True


def _blend(bgr_img: np.ndarray, overlay_rgba: np.ndarray) -> str:
    """Alpha-composites RGBA overlay onto BGR image. Returns base64 PNG."""
    rgb     = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    base    = PILImage.fromarray(rgb).convert("RGBA")
    top     = PILImage.fromarray(overlay_rgba, mode="RGBA")
    blended = PILImage.alpha_composite(base, top).convert("RGB")
    buf     = io.BytesIO()
    blended.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_inference(image_bgr: np.ndarray, is_shell_photo: bool = False) -> dict:
    """
    Runs Mask R-CNN on a BGR numpy image.

    Parameters
    ----------
    image_bgr      : np.ndarray — BGR image from cv2.imdecode
    is_shell_photo : bool — True for Stage 1 shell side photos,
                            False for Stage 2 meat photo.

    Minimum meat pixel threshold:
    ─────────────────────────────
    Shell photos (is_shell_photo=True)  → MIN_MEAT_PX_SHELL_PHOTO = 50
      Low threshold so small exposed meat patches on broken shells
      are not discarded — these are a genuine Grade C indicator.

    Meat photos (is_shell_photo=False)  → MIN_MEAT_PX_MEAT_PHOTO = 500
      High threshold filters nacre false positives from the empty
      shell half in butterfly-style opened mussel photos.

    Single Largest Meat Logic
    ─────────────────────────
    Only the single best meat detection is retained per image.
    All others are discarded as false positives.

    Returns
    -------
    dict:
        class_area      — filtered pixel counts per class
        class_count     — filtered instance counts per class
        mean_score      — mean confidence per class
        total_pixels    — total image pixel count
        overlay_b64     — full 4-class overlay as base64 PNG
        shell_b64       — shell-only overlay as base64 PNG
        bio_b64         — biofouling-only overlay as base64 PNG
        meat_pixels_bgr — BGR pixel values of meat region for color analysis
    """
    # ── Select threshold based on photo type ─────────────────────────────────
    min_meat_px = MIN_MEAT_PX_SHELL_PHOTO if is_shell_photo else MIN_MEAT_PX_MEAT_PHOTO
    print(f"[inference] is_shell_photo={is_shell_photo}  min_meat_px={min_meat_px}")

    predictor = get_predictor()
    outputs   = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")

    pred_classes = instances.pred_classes.numpy()
    scores       = instances.scores.numpy()
    masks        = instances.pred_masks.numpy()

    # ── Determine the single valid meat detection index ───────────────────────
    valid_meat_indices = _keep_largest_meat_only(
        pred_classes, scores, masks, image_bgr, min_meat_px
    )

    # ── Accumulate filtered detections ───────────────────────────────────────
    class_area    = {c: 0.0 for c in CATEGORIES}
    class_score   = {c: []  for c in CATEGORIES}
    class_count   = {c: 0   for c in CATEGORIES}
    meat_mask_all = np.zeros(image_bgr.shape[:2], dtype=bool)

    for i, cls_idx in enumerate(pred_classes):
        cls       = CATEGORIES[cls_idx]
        score     = float(scores[i])
        mask_area = float(masks[i].sum())

        if cls == "meat":
            if i not in valid_meat_indices:
                continue
        else:
            if not _passes_filter(cls, score, mask_area, min_meat_px):
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

    # ── Full combined overlay ─────────────────────────────────────────────────
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        combined = np.zeros((h, w), dtype=bool)
        for i, cls_idx in enumerate(pred_classes):
            if CATEGORIES[cls_idx] != cls:
                continue
            if cls == "meat":
                if i not in valid_meat_indices:
                    continue
            else:
                if not _passes_filter(cls, float(scores[i]), float(masks[i].sum()), min_meat_px):
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
        if not _passes_filter(cls, float(scores[i]), float(masks[i].sum()), min_meat_px):
            continue
        shell_overlay[masks[i].astype(bool)] = (100, 180, 255, 140)
    shell_b64 = _blend(image_bgr, shell_overlay)

    # ── Biofouling-only overlay ───────────────────────────────────────────────
    bio_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for i, cls_idx in enumerate(pred_classes):
        cls = CATEGORIES[cls_idx]
        if cls not in ("attached biofouling", "residual biofouling"):
            continue
        if not _passes_filter(cls, float(scores[i]), float(masks[i].sum()), min_meat_px):
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