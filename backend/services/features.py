import numpy as np
import cv2
import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../weights/rf_config.json")

with open(_CONFIG_PATH) as f:
    CONFIG = json.load(f)

# Calibration
PX_PER_MM   = CONFIG["px_per_mm"]     
TILE_PX     = CONFIG["tile_px"]         
TILE_MM     = CONFIG["tile_mm"]         
PX_PER_MM2  = PX_PER_MM ** 2           

STAGE1_COLS = CONFIG["stage1_cols"]     
STAGE2_COLS = CONFIG["stage2_cols"]     
ALL_COLS    = CONFIG["all_cols"]        

# Score threshold for Mask R-CNN
SCORE_THRESH = CONFIG["score_thresh"]  

CATEGORIES = CONFIG["categories"]

COLOR_THRESH_AB = CONFIG["color_thresh_AB"]   # 8
COLOR_THRESH_BC = CONFIG["color_thresh_BC"]   # 17

RATIO_DISPLAY_HIGH   = 68.0
RATIO_DISPLAY_MEDIUM = 56.0

def _std_lab_to_opencv(lab):
    L, a, b = lab
    return np.array([
        L * 255.0 / 100.0,   # L: 0–100  → 0–255
        a + 128.0,            # a: -128–127 → 0–255
        b + 128.0,            # b: -128–127 → 0–255
    ], dtype=np.float32)


_calib = CONFIG["color_calib"]
GRADE_LAB_REFS = {
    "A": _std_lab_to_opencv(_calib["grade_A_lab"]), 
    "B": _std_lab_to_opencv(_calib["grade_B_lab"]), 
    "C": _std_lab_to_opencv(_calib["grade_C_lab"]),  
}

def px_to_mm2(pixels: float) -> float:
    return round(pixels / PX_PER_MM2, 4)


def px_to_pct(region_px: float, total_px: float) -> float:
    if total_px == 0:
        return 0.0
    return round((region_px / total_px) * 100, 4)

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
    if ratio >= RATIO_DISPLAY_HIGH:
        return "High meat yield"
    elif ratio >= RATIO_DISPLAY_MEDIUM:
        return "Medium meat yield"
    return "Low meat yield"


def meat_yield_weight_approx(ratio: float) -> str:
    if ratio >= RATIO_DISPLAY_HIGH:
        return "above 25%"
    elif ratio >= RATIO_DISPLAY_MEDIUM:
        return "20–25%"
    return "below 20%"


def color_deviation_label(raw_dist: float) -> str:
    if raw_dist <= COLOR_THRESH_AB:
        return "Low Color Deviation"
    elif raw_dist <= COLOR_THRESH_BC:
        return "Moderate Color Deviation"
    return "High Color Deviation"

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

    lab_pixels = cv2.cvtColor(
        pixels.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB
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
    if meat_pixels_bgr is None or len(meat_pixels_bgr) == 0:
        return 50.0, "C", "Unknown", "#cccccc"

    dominant  = _dominant_lab(meat_pixels_bgr)
    hex_color = _mean_rgb_hex(meat_pixels_bgr)

    # Compute DeltaE76 to each grade reference (all in OpenCV Lab scale)
    dist_A = float(np.linalg.norm(dominant - GRADE_LAB_REFS["A"]))
    dist_B = float(np.linalg.norm(dominant - GRADE_LAB_REFS["B"]))
    dist_C = float(np.linalg.norm(dominant - GRADE_LAB_REFS["C"]))

    print(f"[color] dominant_lab(opencv):{dominant.round(1)} "
          f"dist_A:{dist_A:.1f} dist_B:{dist_B:.1f} dist_C:{dist_C:.1f}")

    # Nearest grade reference determines color grade
    min_dist    = min(dist_A, dist_B, dist_C)
    color_grade = "A" if min_dist == dist_A else "B" if min_dist == dist_B else "C"
    raw_dist    = min_dist
    color_label = color_deviation_label(raw_dist)

    print(f"[color] → nearest grade:{color_grade}  "
          f"ΔE:{raw_dist:.1f}  label:{color_label}")

    return raw_dist, color_grade, color_label, hex_color

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


def extract_meat_features(inference_result: dict, shell_mm2_from_side: float) -> dict:
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
        "meat_mm2":                  px_to_mm2(meat_px),
        "meat_shell_ratio":          raw_ratio,
        "flesh_color_dev":           round(raw_dist, 4),
        "conf_meat":                 round(ms["meat"], 4),
        "meat_ratio_display":        min(raw_ratio, 100.0),
        "meat_yield_label":          meat_yield_label(raw_ratio),
        "meat_yield_weight_approx":  meat_yield_weight_approx(raw_ratio),
        "flesh_color_grade":         color_grade,
        "flesh_color_label":         color_label,
        "flesh_color_dev_label":     color_deviation_label(raw_dist),
        "flesh_color_hex":           hex_color,
    }

    return features

def build_stage1_vector(combined_side_features: dict) -> list:
    return [combined_side_features[c] for c in STAGE1_COLS]


def build_all_vector(combined_side_features: dict, meat_features: dict) -> list:
    merged = {**combined_side_features, **meat_features}
    return [merged[c] for c in ALL_COLS]

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