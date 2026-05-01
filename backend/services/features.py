"""
features.py
===========
Feature extraction pipeline for the Green Mussel Quality Assessment System.
All calibration values, thresholds, and column orders are loaded from
rf_config.json to guarantee they match the RF training pipeline exactly.
"""

import numpy as np
import cv2
import json
import os

# ── Load config ───────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../weights/rf_config.json")

with open(_CONFIG_PATH) as f:
    CONFIG = json.load(f)

# Calibration
PX_PER_MM   = CONFIG["px_per_mm"]       # 17.062992125984252
TILE_PX     = CONFIG["tile_px"]         # 433.4
TILE_MM     = CONFIG["tile_mm"]         # 25.4
PX_PER_MM2  = PX_PER_MM ** 2           # 291.146...

# Column orders (must match RF training exactly)
STAGE1_COLS = CONFIG["stage1_cols"]     # 13 features
STAGE2_COLS = CONFIG["stage2_cols"]     # 4 features
ALL_COLS    = CONFIG["all_cols"]        # 17 features

# Score threshold for Mask R-CNN
SCORE_THRESH = CONFIG["score_thresh"]   # 0.3

# Category names (must match Detectron2 training order)
CATEGORIES = CONFIG["categories"]       # ['shell','meat','residual biofouling','attached biofouling']

# Color thresholds (from empirical calibration, 45 images)
COLOR_THRESH_AB = CONFIG["color_thresh_AB"]   # 8
COLOR_THRESH_BC = CONFIG["color_thresh_BC"]   # 17

# Shell-to-meat ratio thresholds
RATIO_THRESH_AB = CONFIG["ratio_thresh_AB"]   # 65
RATIO_THRESH_BC = CONFIG["ratio_thresh_BC"]   # 50

# CIE Lab reference colors per grade
_calib = CONFIG["color_calib"]
GRADE_LAB_REFS = {
    "A": np.array(_calib["grade_A_lab"]),   # [34.7, 11.4, 15.2]
    "B": np.array(_calib["grade_B_lab"]),   # [28.2,  6.6, 12.2]
    "C": np.array(_calib["grade_C_lab"]),   # [35.3, 10.4, 14.4]
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Unit conversion helpers
# ═════════════════════════════════════════════════════════════════════════════

def px_to_mm2(pixels: float) -> float:
    """Convert pixel area → mm² using config calibration."""
    return round(pixels / PX_PER_MM2, 4)


def px_to_pct(region_px: float, total_px: float) -> float:
    """Convert pixel region → % of total area. Returns 0.0 on zero division."""
    if total_px == 0:
        return 0.0
    return round((region_px / total_px) * 100, 4)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Derived feature calculations  (match training notebook exactly)
# ═════════════════════════════════════════════════════════════════════════════

def compute_attach_ratio(bio_attach_px: float, shell_px: float) -> float:
    if shell_px == 0:
        return 0.0
    return round(bio_attach_px / shell_px, 4)


def compute_bio_coverage_pct(bio_attach_px: float,
                              bio_resid_px: float,
                              shell_px: float) -> float:
    total_bio = bio_attach_px + bio_resid_px
    return px_to_pct(total_bio, shell_px)


def compute_meat_shell_ratio(meat_px: float, shell_px: float) -> float:
    """Meat area as % of shell area — capped at 100% to prevent false positives."""
    if shell_px == 0:
        return 0.0
    ratio = round((meat_px / shell_px) * 100, 4)
    return min(ratio, 100.0)   # ← cap at 100%


def compute_bio_level(bio_coverage_pct: float) -> int:
    """
    Ordinal encoding of biofouling severity matching training data.
      0 = clean     (0–10%)
      1 = light     (11–30%)
      2 = moderate  (31–50%)
      3 = heavy     (>50%)
    """
    if bio_coverage_pct <= 10:
        return 0
    elif bio_coverage_pct <= 30:
        return 1
    elif bio_coverage_pct <= 50:
        return 2
    return 3


# ═════════════════════════════════════════════════════════════════════════════
# 3. Color deviation (CIE Lab DeltaE76)
# ═════════════════════════════════════════════════════════════════════════════

# Orange and cream reference swatches (same as training notebook)
_ORANGE_RGB = np.array([
    [160,  80,  30], [140,  60,  20], [120,  45,  10],
    [170,  90,  40], [110,  35,   5], [183, 121,  65],
    [165, 100,  50], [155,  85,  40], [145,  70,  30],
    [130,  55,  20], [110,   1,   0], [100,  20,  10],
    [ 90,  10,   5], [220, 160, 100], [210, 150,  90],
    [230, 170, 110], [200, 140,  85], [215, 155,  95],
    [225, 165, 105], [205, 145,  88], [235, 175, 115],
    [195, 135,  80], [240, 185, 120],
], dtype=np.uint8)

_CREAM_RGB = np.array([
    [213, 183, 143], [230, 210, 170], [200, 170, 130],
    [220, 200, 160], [240, 220, 185], [210, 185, 150],
    [225, 195, 155], [195, 165, 125], [235, 205, 165],
    [205, 175, 135], [245, 235, 215], [250, 240, 225],
    [238, 228, 210], [185, 160, 120], [175, 150, 110],
    [190, 165, 128],
], dtype=np.uint8)


def _rgb_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    lab_list = []
    for c in rgb_array:
        lab = cv2.cvtColor(
            c.reshape(1, 1, 3).astype(np.uint8),
            cv2.COLOR_RGB2LAB
        ).reshape(3)
        lab_list.append(lab)
    return np.array(lab_list, dtype=np.float32)


_STANDARD_LAB = {
    "orange": _rgb_to_lab(_ORANGE_RGB),
    "cream":  _rgb_to_lab(_CREAM_RGB),
}


def _dominant_lab(meat_pixels_bgr: np.ndarray) -> np.ndarray:
    """Find dominant color cluster using k-means (k=3). Returns largest cluster center in LAB."""
    from sklearn.cluster import MiniBatchKMeans

    if len(meat_pixels_bgr) < 10:
        lab = cv2.cvtColor(
            meat_pixels_bgr.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)
        return lab.mean(axis=0)

    pixels = meat_pixels_bgr
    if len(pixels) > 5000:
        idx    = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

    pixels_rgb = pixels[:, ::-1]  # BGR → RGB
    lab_pixels = cv2.cvtColor(
        pixels_rgb.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)

    try:
        kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, n_init=3)
        labels = kmeans.fit_predict(lab_pixels)
        dominant_idx = int(np.argmax(np.bincount(labels)))
        return kmeans.cluster_centers_[dominant_idx]
    except Exception:
        return lab_pixels.mean(axis=0)


def _mean_rgb_hex(meat_pixels_bgr: np.ndarray) -> str:
    mean_bgr = meat_pixels_bgr.mean(axis=0)
    r, g, b  = int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])
    return f"#{r:02x}{g:02x}{b:02x}"


def color_deviation(meat_pixels_bgr: np.ndarray) -> tuple:
    """
    Classifies flesh color as orange or cream, computes LAB deviation
    from reference swatches using thresholds from rf_config.json.

    Returns:
        (raw_dist, color_grade, color_label, hex_color)
        raw_dist    — ΔE value fed into RF as flesh_color_dev
        color_grade — 'A' if raw_dist <= thresh_AB, 'B' if <= thresh_BC, else 'C'
        color_label — 'Orange' or 'Cream'
        hex_color   — mean BGR hex for display
    """
    if meat_pixels_bgr is None or len(meat_pixels_bgr) == 0:
        return 50.0, "B", "Unknown", "#cccccc"

    dominant  = _dominant_lab(meat_pixels_bgr)
    hex_color = _mean_rgb_hex(meat_pixels_bgr)

    dist_orange = float(min(np.linalg.norm(dominant - r) for r in _STANDARD_LAB["orange"]))
    dist_cream  = float(min(np.linalg.norm(dominant - r) for r in _STANDARD_LAB["cream"]))

    if dist_orange <= dist_cream:
        raw_dist    = dist_orange
        color_label = "Orange"
    else:
        raw_dist    = dist_cream
        color_label = "Cream"

    # Use thresholds from config
    if raw_dist <= COLOR_THRESH_AB:
        color_grade = "A"
    elif raw_dist <= COLOR_THRESH_BC:
        color_grade = "B"
    else:
        color_grade = "C"

    print(f"[color] dominant_lab:{dominant.round(1)} "
          f"dist_orange:{dist_orange:.1f} dist_cream:{dist_cream:.1f} "
          f"→ {color_label} ΔE:{raw_dist:.1f} grade:{color_grade}")

    return raw_dist, color_grade, color_label, hex_color


# ═════════════════════════════════════════════════════════════════════════════
# 4. Stage 1 — shell side features
# ═════════════════════════════════════════════════════════════════════════════

def extract_side_features(inference_result: dict) -> dict:
    """
    Converts Mask R-CNN inference output → Stage 1 feature dict.
    Uses feature_conversion.py logic with rf_config.json calibration.
    """
    ca = inference_result["class_area"]
    cc = inference_result["class_count"]
    ms = inference_result["mean_score"]

    shell_px      = ca["shell"]
    bio_attach_px = ca["attached biofouling"]
    bio_resid_px  = ca["residual biofouling"]
    side_meat_px  = ca["meat"]

    bio_coverage = compute_bio_coverage_pct(bio_attach_px, bio_resid_px, shell_px)

    features = {
        # mm² areas
        "shell_mm2":        px_to_mm2(shell_px),
        "bio_attach_mm2":   px_to_mm2(bio_attach_px),
        "bio_resid_mm2":    px_to_mm2(bio_resid_px),
        "side_meat_mm2":    px_to_mm2(side_meat_px),

        # Derived ratios / percentages
        "bio_coverage_pct": bio_coverage,
        "attach_ratio":     compute_attach_ratio(bio_attach_px, shell_px),
        "side_meat_pct":    px_to_pct(side_meat_px, shell_px),
        "bio_level":        float(compute_bio_level(bio_coverage)),

        # Instance counts
        "n_bio_attach":     float(cc["attached biofouling"]),
        "n_bio_resid":      float(cc["residual biofouling"]),

        # Confidence scores
        "conf_shell_side":  round(ms["shell"], 4),
        "conf_bio_attach":  round(ms["attached biofouling"], 4),
        "conf_bio_resid":   round(ms["residual biofouling"], 4),
    }

    # Validate all stage1 columns are present
    missing = [c for c in STAGE1_COLS if c not in features]
    if missing:
        raise ValueError(f"Missing Stage 1 features: {missing}")

    return features


def average_side_features(feat1: dict, feat2: dict) -> dict:
    """
    Combines Side A + Side B features.
      bio_level     → MAX  (worst side counts)
      side_meat_pct → MAX  (broken shell signal)
      side_meat_mm2 → SUM  (total exposed meat)
      all others    → mean
    """
    combined = {}
    for key in feat1:
        if key in ("bio_level", "side_meat_pct"):
            combined[key] = max(feat1[key], feat2[key])
        elif key == "side_meat_mm2":
            combined[key] = feat1[key] + feat2[key]
        else:
            combined[key] = (feat1[key] + feat2[key]) / 2.0
    return combined


def is_broken_shell(combined_side_features: dict) -> bool:
    """Returns True if meat is visible from either shell side view (cracked/open shell)."""
    return combined_side_features["side_meat_pct"] > 0


# ═════════════════════════════════════════════════════════════════════════════
# 5. Stage 2 — meat image features
# ═════════════════════════════════════════════════════════════════════════════

def extract_meat_features(inference_result: dict, shell_mm2_from_side: float) -> dict:
    """
    Converts meat image inference output → Stage 2 feature dict.
    meat_shell_ratio uses px_to_pct(meat_px, shell_px) matching training.
    """
    ca = inference_result["class_area"]
    ms = inference_result["mean_score"]

    meat_px  = ca["meat"]
    # Convert shell_mm2 back to px for ratio calculation
    shell_px = shell_mm2_from_side * PX_PER_MM2

    raw_dist, color_grade, color_label, hex_color = color_deviation(
        inference_result["meat_pixels_bgr"]
    )

    features = {
        "meat_mm2":          px_to_mm2(meat_px),
        "meat_shell_ratio":  compute_meat_shell_ratio(meat_px, shell_px),
        "flesh_color_dev":   round(raw_dist, 4),
        "conf_meat":         round(ms["meat"], 4),
        # Display-only fields (not fed to RF)
        "flesh_color_grade": color_grade,
        "flesh_color_label": color_label,
        "flesh_color_hex":   hex_color,
    }

    return features


# ═════════════════════════════════════════════════════════════════════════════
# 6. Build model input arrays
# ═════════════════════════════════════════════════════════════════════════════

def build_stage1_vector(combined_side_features: dict) -> list:
    """Returns 13 values in STAGE1_COLS order — fed to rf_stage1_shell.pkl."""
    return [combined_side_features[c] for c in STAGE1_COLS]


def build_all_vector(combined_side_features: dict, meat_features: dict) -> list:
    """Returns 17 values in ALL_COLS order — fed to rf_stage2_final.pkl."""
    merged = {**combined_side_features, **meat_features}
    return [merged[c] for c in ALL_COLS]


# ═════════════════════════════════════════════════════════════════════════════
# 7. Range validation (catches bad images / segmentation failures)
# ═════════════════════════════════════════════════════════════════════════════

def validate_ranges(features: dict) -> list:
    """
    Checks features are within expected ranges.
    Returns list of warning strings — empty list = all OK.
    """
    checks = {
        "shell_mm2":        (50,   2000),
        "bio_coverage_pct": (0,    100),
        "meat_shell_ratio": (0,    100),
        "flesh_color_dev":  (0,    50),
        "conf_shell_side":  (0.5,  1.0),
        "conf_meat":        (0.5,  1.0),
    }
    warnings = []
    for feat, (lo, hi) in checks.items():
        if feat in features:
            val = features[feat]
            if not (lo <= val <= hi):
                warnings.append(
                    f"WARNING: {feat}={val:.3f} outside expected range [{lo}, {hi}]"
                )
    return warnings