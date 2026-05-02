"""
features.py
===========
Feature extraction pipeline for the Green Mussel Quality Assessment System.
All calibration values, thresholds, and column orders are loaded from
rf_config.json to guarantee they match the RF training pipeline exactly.

meat_shell_ratio stats from mussel_features_183.csv:
  mean=62.5  std=14.3  min=37.1  max=157.5
  Grade A mean=68.1  |  Grade B mean=62.6  |  Grade C mean=56.6
  Thresholds: High >= 68, Medium >= 56, Low < 56

FIX (meat_shell_ratio denominator):
  Removed × 2 from shell_px to match training notebook exactly:
    shell_px = shell_mm2_from_side * PX_PER_MM2   (no × 2)

NOTE (display ratio):
  The pixel-based ratio is shown as-is — no ÷2 scaling applied.
  The RF receives raw_ratio exactly as trained.
  meat_yield_weight_approx maps the pixel ratio to an approximate
  weight-based range from the grading table, shown for reference only.
  This mapping is not a validated physical conversion — it is an
  indicative label based on the study's grading table thresholds.
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
CATEGORIES = CONFIG["categories"]

# Color thresholds (from empirical calibration, 45 images)
COLOR_THRESH_AB = CONFIG["color_thresh_AB"]   # 8
COLOR_THRESH_BC = CONFIG["color_thresh_BC"]   # 17

# Shell-to-meat ratio display thresholds (pixel scale from training data)
#   Grade A mean = 68.1%  →  High   >= 68
#   Grade B mean = 62.6%  →  Medium >= 56
#   Grade C mean = 56.6%  →  Low    <  56
RATIO_DISPLAY_HIGH   = 68.0
RATIO_DISPLAY_MEDIUM = 56.0

# CIE Lab reference colors per grade
_calib = CONFIG["color_calib"]
GRADE_LAB_REFS = {
    "A": np.array(_calib["grade_A_lab"]),
    "B": np.array(_calib["grade_B_lab"]),
    "C": np.array(_calib["grade_C_lab"]),
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Unit conversion helpers
# ═════════════════════════════════════════════════════════════════════════════

def px_to_mm2(pixels: float) -> float:
    return round(pixels / PX_PER_MM2, 4)


def px_to_pct(region_px: float, total_px: float) -> float:
    if total_px == 0:
        return 0.0
    return round((region_px / total_px) * 100, 4)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Derived feature calculations
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
    """
    Meat area as % of shell area — pixel-based meat yield proxy.
    Training data range: 37–157%, mean ~62%.
    Fed to RF exactly as-is — do NOT modify.
    """
    return px_to_pct(meat_px, shell_px)


def compute_bio_level(bio_coverage_pct: float) -> int:
    if bio_coverage_pct <= 10:
        return 0
    elif bio_coverage_pct <= 30:
        return 1
    elif bio_coverage_pct <= 50:
        return 2
    return 3


def meat_yield_label(ratio: float) -> str:
    """
    Pixel-based meat yield label aligned to training data distribution.
      >= 68% → High meat yield   (Grade A range)
      >= 56% → Medium meat yield (Grade B range)
      <  56% → Low meat yield   (Grade C range)
    """
    if ratio >= RATIO_DISPLAY_HIGH:
        return "High meat yield"
    elif ratio >= RATIO_DISPLAY_MEDIUM:
        return "Medium meat yield"
    return "Low meat yield"


def meat_yield_weight_approx(ratio: float) -> str:
    """
    Maps pixel ratio to approximate weight-based range from the grading table.
    Shown in UI for reference only — not a validated physical conversion.
      Pixel >= 68%  →  est. weight yield: above 25%  (Grade A)
      Pixel >= 56%  →  est. weight yield: 20–25%     (Grade B)
      Pixel <  56%  →  est. weight yield: below 20%  (Grade C)
    """
    if ratio >= RATIO_DISPLAY_HIGH:
        return "above 25%"
    elif ratio >= RATIO_DISPLAY_MEDIUM:
        return "20–25%"
    return "below 20%"


def color_deviation_label(raw_dist: float) -> str:
    """
    Color deviation severity label aligned to grading table.
      <= COLOR_THRESH_AB (8)  → Low Color Deviation
      <= COLOR_THRESH_BC (17) → Moderate Color Deviation
      >  COLOR_THRESH_BC      → High Color Deviation
    """
    if raw_dist <= COLOR_THRESH_AB:
        return "Low Color Deviation"
    elif raw_dist <= COLOR_THRESH_BC:
        return "Moderate Color Deviation"
    return "High Color Deviation"


# ═════════════════════════════════════════════════════════════════════════════
# 3. Color deviation (CIE Lab DeltaE76)
# ═════════════════════════════════════════════════════════════════════════════

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

    pixels_rgb = pixels[:, ::-1]
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
    Classifies flesh color as orange or cream, computes LAB deviation.
    Returns: (raw_dist, color_grade, color_label, hex_color)
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
    ca = inference_result["class_area"]
    cc = inference_result["class_count"]
    ms = inference_result["mean_score"]

    shell_px      = ca["shell"]
    bio_attach_px = ca["attached biofouling"]
    bio_resid_px  = ca["residual biofouling"]
    side_meat_px  = ca["meat"]

    bio_coverage = compute_bio_coverage_pct(bio_attach_px, bio_resid_px, shell_px)

    features = {
        "shell_mm2":        px_to_mm2(shell_px),
        "bio_attach_mm2":   px_to_mm2(bio_attach_px),
        "bio_resid_mm2":    px_to_mm2(bio_resid_px),
        "side_meat_mm2":    px_to_mm2(side_meat_px),
        "bio_coverage_pct": bio_coverage,
        "attach_ratio":     compute_attach_ratio(bio_attach_px, shell_px),
        "side_meat_pct":    px_to_pct(side_meat_px, shell_px),
        "bio_level":        float(compute_bio_level(bio_coverage)),
        "n_bio_attach":     float(cc["attached biofouling"]),
        "n_bio_resid":      float(cc["residual biofouling"]),
        "conf_shell_side":  round(ms["shell"], 4),
        "conf_bio_attach":  round(ms["attached biofouling"], 4),
        "conf_bio_resid":   round(ms["residual biofouling"], 4),
    }

    missing = [c for c in STAGE1_COLS if c not in features]
    if missing:
        raise ValueError(f"Missing Stage 1 features: {missing}")

    return features


def average_side_features(feat1: dict, feat2: dict) -> dict:
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
    return combined_side_features["side_meat_pct"] > 0


# ═════════════════════════════════════════════════════════════════════════════
# 5. Stage 2 — meat image features
# ═════════════════════════════════════════════════════════════════════════════

def extract_meat_features(inference_result: dict, shell_mm2_from_side: float) -> dict:
    """
    Converts meat image inference output → Stage 2 feature dict.

    Denominator: shell_px = shell_mm2_from_side * PX_PER_MM2  (no × 2)
    Matches training notebook exactly.

    Display: raw pixel ratio shown as-is, capped at 100 for UI.
    meat_yield_weight_approx provides an indicative weight-based range
    from the grading table — for reference only, not a validated conversion.
    """
    ca = inference_result["class_area"]
    ms = inference_result["mean_score"]

    meat_px  = ca["meat"]
    shell_px = shell_mm2_from_side * PX_PER_MM2

    print(f"[meat_shell_ratio] "
          f"shell_mm2={shell_mm2_from_side:.1f}  "
          f"shell_px={shell_px:.0f}  "
          f"meat_px={meat_px:.0f}  "
          f"ratio={meat_px / shell_px * 100:.1f}%"
          if shell_px > 0 else
          "[meat_shell_ratio] shell_px=0 — cannot compute ratio")

    raw_ratio = compute_meat_shell_ratio(meat_px, shell_px)

    raw_dist, color_grade, color_label, hex_color = color_deviation(
        inference_result["meat_pixels_bgr"]
    )

    features = {
        # ── Fed to RF (must match training data exactly) ──────────────────
        "meat_mm2":                  px_to_mm2(meat_px),
        "meat_shell_ratio":          raw_ratio,
        "flesh_color_dev":           round(raw_dist, 4),
        "conf_meat":                 round(ms["meat"], 4),

        # ── Display-only fields (NOT fed to RF) ───────────────────────────
        "meat_ratio_display":        min(raw_ratio, 100.0),
        "meat_yield_label":          meat_yield_label(raw_ratio),
        "meat_yield_weight_approx":  meat_yield_weight_approx(raw_ratio),
        "flesh_color_grade":         color_grade,
        "flesh_color_label":         color_label,
        "flesh_color_dev_label":     color_deviation_label(raw_dist),
        "flesh_color_hex":           hex_color,
    }

    return features


# ═════════════════════════════════════════════════════════════════════════════
# 6. Build model input arrays
# ═════════════════════════════════════════════════════════════════════════════

def build_stage1_vector(combined_side_features: dict) -> list:
    return [combined_side_features[c] for c in STAGE1_COLS]


def build_all_vector(combined_side_features: dict, meat_features: dict) -> list:
    merged = {**combined_side_features, **meat_features}
    return [merged[c] for c in ALL_COLS]


# ═════════════════════════════════════════════════════════════════════════════
# 7. Range validation
# ═════════════════════════════════════════════════════════════════════════════

def validate_ranges(features: dict) -> list:
    checks = {
        "shell_mm2":        (50,    2000),
        "bio_coverage_pct": (0,     100),
        "meat_shell_ratio": (0,     200),
        "flesh_color_dev":  (0,     50),
        "conf_shell_side":  (0.3,   1.0),
        "conf_meat":        (0.3,   1.0),
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