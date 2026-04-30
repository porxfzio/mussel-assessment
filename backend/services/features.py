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


# ── CIE Lab reference swatches (from training notebook) ─────────
import cv2

def _rgb_to_lab(rgb_array):
    lab_list = []
    for c in rgb_array:
        lab = cv2.cvtColor(
            c.reshape(1, 1, 3).astype(np.uint8),
            cv2.COLOR_RGB2LAB
        ).reshape(3)
        lab_list.append(lab)
    return np.array(lab_list, dtype=np.float32)

ORANGE_RGB = np.array([
    [160,  80,  30], [140,  60,  20], [120,  45,  10],
    [170,  90,  40], [110,  35,   5], [183, 121,  65],
    [165, 100,  50], [155,  85,  40], [145,  70,  30],
    [130,  55,  20], [110,   1,   0], [100,  20,  10],
    [ 90,  10,   5], [220, 160, 100], [210, 150,  90],
    [230, 170, 110], [200, 140,  85], [215, 155,  95],
    [225, 165, 105], [205, 145,  88], [235, 175, 115],
    [195, 135,  80], [240, 185, 120],
], dtype=np.uint8)

CREAM_RGB = np.array([
    [213, 183, 143], [230, 210, 170], [200, 170, 130],
    [220, 200, 160], [240, 220, 185], [210, 185, 150],
    [225, 195, 155], [195, 165, 125], [235, 205, 165],
    [205, 175, 135], [245, 235, 215], [250, 240, 225],
    [238, 228, 210], [185, 160, 120], [175, 150, 110],
    [190, 165, 128],
], dtype=np.uint8)

STANDARD_LAB = {
    "orange": _rgb_to_lab(ORANGE_RGB),
    "cream":  _rgb_to_lab(CREAM_RGB),
}

# Thresholds from notebook
COLOR_THRESH_A  = 40.0
COLOR_THRESH_B  = 80.0


def _dominant_lab(meat_pixels_bgr: np.ndarray) -> np.ndarray:
    """
    Finds the dominant color cluster in meat pixels using k-means (k=3).
    Returns the largest cluster center in LAB space.
    Falls back to mean if k-means fails.
    """
    from sklearn.cluster import MiniBatchKMeans

    if len(meat_pixels_bgr) < 10:
        # Too few pixels — use mean
        lab = cv2.cvtColor(
            meat_pixels_bgr.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)
        return lab.mean(axis=0)

    # Sample max 5000 pixels for speed
    pixels = meat_pixels_bgr
    if len(pixels) > 5000:
        idx    = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

    # Convert BGR → RGB → LAB
    pixels_rgb = pixels[:, ::-1]  # BGR to RGB
    lab_pixels = cv2.cvtColor(
        pixels_rgb.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)

    try:
        kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, n_init=3)
        labels = kmeans.fit_predict(lab_pixels)
        # Pick the largest cluster
        counts  = np.bincount(labels)
        dominant_idx = int(np.argmax(counts))
        return kmeans.cluster_centers_[dominant_idx]
    except Exception:
        return lab_pixels.mean(axis=0)


def _compute_deviation(dominant_lab: np.ndarray, color_class: str) -> float:
    """Min LAB distance from dominant cluster to reference swatches."""
    refs = STANDARD_LAB[color_class]
    return float(min(np.linalg.norm(dominant_lab - r) for r in refs))


def _assign_class(raw_dist: float) -> str:
    if raw_dist <= COLOR_THRESH_A:
        return "A"
    elif raw_dist <= COLOR_THRESH_B:
        return "B"
    else:
        return "C"


def _mean_rgb_hex(meat_pixels_bgr: np.ndarray) -> str:
    """Returns the mean color of meat pixels as a hex string."""
    mean_bgr = meat_pixels_bgr.mean(axis=0)
    r, g, b  = int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])
    return f"#{r:02x}{g:02x}{b:02x}"


def color_deviation(meat_pixels_bgr: np.ndarray) -> tuple:
    """
    Classifies flesh color as orange (female) or cream (male),
    computes LAB deviation from reference swatches, and assigns grade.

    Returns:
        (raw_dist, color_grade, color_label, hex_color)
    """
    if meat_pixels_bgr is None or len(meat_pixels_bgr) == 0:
        return 50.0, "B", "Unknown", "#cccccc"

    dominant = _dominant_lab(meat_pixels_bgr)
    hex_color = _mean_rgb_hex(meat_pixels_bgr)

    # Classify as orange or cream — whichever is closer
    dist_orange = _compute_deviation(dominant, "orange")
    dist_cream  = _compute_deviation(dominant, "cream")

    if dist_orange <= dist_cream:
        classified  = "orange"
        raw_dist    = dist_orange
        color_label = "Orange"
    else:
        classified  = "cream"
        raw_dist    = dist_cream
        color_label = "Cream"

    color_grade = _assign_class(raw_dist)

    print(f"[color] dominant_lab:{dominant.round(1)} "
          f"dist_orange:{dist_orange:.1f} dist_cream:{dist_cream:.1f} "
          f"→ {color_label} ΔE:{raw_dist:.1f} grade:{color_grade}")

    return raw_dist, color_grade, color_label, hex_color


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

    bio_pct = min((total_bio / shell_mm2 * 100) if shell_mm2 > 0 else 0.0, 100.0)
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
    ca = inference_result["class_area"]
    ms = inference_result["mean_score"]

    meat_mm2         = ca['meat'] / PX_PER_MM2
    meat_shell_ratio = (meat_mm2 / shell_mm2_from_side * 100) \
                       if shell_mm2_from_side > 0 else 0.0

    raw_dist, color_grade, color_label, hex_color = color_deviation(
        inference_result["meat_pixels_bgr"]
    )

    return {
        'meat_mm2':           meat_mm2,
        'meat_shell_ratio':   meat_shell_ratio,
        'flesh_color_dev':    raw_dist,        # raw ΔE — what RF uses
        'conf_meat':          ms['meat'],
        # Display fields
        'flesh_color_grade':  color_grade,     # A / B / C
        'flesh_color_label':  color_label,     # Orange (Female) / Cream (Male)
        'flesh_color_hex':    hex_color,       # actual detected color
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