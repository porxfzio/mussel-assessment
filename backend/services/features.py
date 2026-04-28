# ── feature.py ──────────────────────────────────────────────────────

import numpy as np
import cv2

# ── Scale calibration ──────────────────────────────────────────────────────
# Measured from your tile calibration (10 measurements, mean = 433.45 px)
# 1 inch tile = 25.4 mm  →  PX_PER_MM = 433.45 / 25.4 = 17.065
# These must match what you used during feature extraction.
TILE_MM    = 25.4
TILE_PX    = 433.45          # ← replace with your exact mean from Cell 4 output
PX_PER_MM  = TILE_PX / TILE_MM
PX_PER_MM2 = PX_PER_MM ** 2

CATEGORIES = ['shell', 'meat', 'residual biofouling', 'attached biofouling']

# ── CIE Lab reference colors (empirically measured, 45 images, 15/grade) ──
FRESH_COLOR_REFS_LAB = [
    np.array([34.7, 11.4, 15.2]),   # Grade A flesh
    np.array([28.2,  6.6, 12.2]),   # Grade B flesh
    np.array([35.3, 10.4, 14.4]),   # Grade C flesh
]

# ── Feature column order — must match RF training exactly ─────────────────
STAGE1_COLS = [
    'shell_mm2', 'bio_attach_mm2', 'bio_resid_mm2',
    'bio_coverage_pct', 'attach_ratio', 'bio_level',
    'conf_shell_side', 'conf_bio_attach', 'conf_bio_resid',
    'n_bio_attach', 'n_bio_resid',
    'side_meat_mm2', 'side_meat_pct',
]
STAGE2_COLS = [
    'meat_mm2', 'meat_shell_ratio',
    'flesh_color_dev', 'conf_meat',
]
ALL_COLS = STAGE1_COLS + STAGE2_COLS


def color_deviation(meat_pixels_bgr: np.ndarray) -> float:
    """
    CIE Lab DeltaE76 distance from empirically measured fresh mussel flesh.
    Returns normalised score 0-100.  Returns 50.0 if no meat detected.

    Calibrated thresholds:
      score <= 8  -> Grade A  (DeltaE76 <= 4.0)
      score <= 17 -> Grade B  (DeltaE76 <= 8.5)
      score >  17 -> Grade C  (DeltaE76 >  8.5)
    """
    if meat_pixels_bgr is None or len(meat_pixels_bgr) == 0:
        return 50.0

    lab_cv2 = cv2.cvtColor(
        meat_pixels_bgr.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(float)

    mean_lab = np.array([
        lab_cv2[:, 0].mean() * 100.0 / 255.0,
        lab_cv2[:, 1].mean() - 128.0,
        lab_cv2[:, 2].mean() - 128.0,
    ])

    delta_e = float(min(
        np.linalg.norm(mean_lab - ref) for ref in FRESH_COLOR_REFS_LAB
    ))

    return float(np.clip(delta_e * 2.0, 0.0, 100.0))


def extract_side_features(inference_result: dict) -> dict:
    """
    Compute Stage 1 features from one side image's inference result.
    Mirrors extract_side_features() in your Feature Extraction notebook.
    """
    ca = inference_result["class_area"]
    cc = inference_result["class_count"]
    ms = inference_result["mean_score"]

    shell_mm2   = ca['shell']               / PX_PER_MM2
    bio_att_mm2 = ca['attached biofouling'] / PX_PER_MM2
    bio_res_mm2 = ca['residual biofouling'] / PX_PER_MM2
    total_bio   = bio_att_mm2 + bio_res_mm2

    bio_pct      = (total_bio / shell_mm2 * 100) if shell_mm2 > 0 else 0.0
    attach_ratio = (bio_att_mm2 / total_bio)      if total_bio > 0 else 0.0

    if   bio_pct <= 5:   bio_level = 0
    elif bio_pct <= 25:  bio_level = 1
    elif bio_pct <= 60:  bio_level = 2
    else:                bio_level = 3

    side_meat_mm2 = ca['meat'] / PX_PER_MM2
    visible_mm2   = shell_mm2 + side_meat_mm2
    side_meat_pct = (side_meat_mm2 / visible_mm2 * 100) if visible_mm2 > 0 else 0.0

    return {
        'shell_mm2'       : shell_mm2,
        'bio_attach_mm2'  : bio_att_mm2,
        'bio_resid_mm2'   : bio_res_mm2,
        'bio_coverage_pct': bio_pct,
        'attach_ratio'    : attach_ratio,
        'bio_level'       : float(bio_level),
        'conf_shell_side' : ms['shell'],
        'conf_bio_attach' : ms['attached biofouling'],
        'conf_bio_resid'  : ms['residual biofouling'],
        'n_bio_attach'    : float(cc['attached biofouling']),
        'n_bio_resid'     : float(cc['residual biofouling']),
        'side_meat_mm2'   : side_meat_mm2,
        'side_meat_pct'   : side_meat_pct,
    }


def average_side_features(feat1: dict, feat2: dict) -> dict:
    """
    Combines firstside + secondside.
    bio_level    → MAX (worst side counts)
    side_meat_pct → MAX (either side showing meat is a broken shell signal)
    side_meat_mm2 → SUM (total exposed meat across both sides)
    all others   → mean
    """
    combined = {}
    for key in feat1:
        if key == 'bio_level':
            combined[key] = max(feat1[key], feat2[key])
        elif key == 'side_meat_pct':
            combined[key] = max(feat1[key], feat2[key])
        elif key == 'side_meat_mm2':
            combined[key] = feat1[key] + feat2[key]
        else:
            combined[key] = (feat1[key] + feat2[key]) / 2.0
    return combined


def extract_meat_features(inference_result: dict, shell_mm2_from_side: float) -> dict:
    """
    Compute Stage 2 features from the meat image inference result.
    Mirrors extract_meat_features() in your Feature Extraction notebook.
    """
    ca = inference_result["class_area"]
    ms = inference_result["mean_score"]

    meat_mm2 = ca['meat'] / PX_PER_MM2
    meat_shell_ratio = (meat_mm2 / shell_mm2_from_side * 100) \
                       if shell_mm2_from_side > 0 else 0.0

    flesh_color_dev = color_deviation(inference_result["meat_pixels_bgr"])

    return {
        'meat_mm2'        : meat_mm2,
        'meat_shell_ratio': meat_shell_ratio,
        'flesh_color_dev' : flesh_color_dev,
        'conf_meat'       : ms['meat'],
    }


def build_stage1_vector(combined_side_features: dict) -> list:
    """Returns feature values in the exact column order STAGE1_COLS."""
    return [combined_side_features[c] for c in STAGE1_COLS]


def build_all_vector(combined_side_features: dict, meat_features: dict) -> list:
    """Returns feature values in the exact column order ALL_COLS."""
    merged = {**combined_side_features, **meat_features}
    return [merged[c] for c in ALL_COLS]


def is_broken_shell(combined_side_features: dict) -> bool:
    """
    Returns True if side_meat_pct > 0 on the worst side —
    exposed meat on a shell view means the shell is cracked/open.
    """
    return combined_side_features['side_meat_pct'] > 0