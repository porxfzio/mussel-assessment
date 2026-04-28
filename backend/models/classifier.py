import joblib
import numpy as np
import os

_rf_stage1 = None
_rf_stage2 = None
_le        = None

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "../weights")

def _load_models():
    global _rf_stage1, _rf_stage2, _le
    if _rf_stage1 is None:
        _rf_stage1 = joblib.load(os.path.join(WEIGHTS_DIR, "rf_stage1_shell.pkl"))
        _rf_stage2 = joblib.load(os.path.join(WEIGHTS_DIR, "rf_stage2_final.pkl"))
        _le        = joblib.load(os.path.join(WEIGHTS_DIR, "label_encoder.pkl"))


def predict_initial_grade(stage1_vector: list) -> dict:
    # No pkl needed — purely rule-based for Stage 1
    from services.features import STAGE1_COLS
    feat    = dict(zip(STAGE1_COLS, stage1_vector))
    bio_pct = feat['bio_coverage_pct']
    broken  = feat['side_meat_pct'] > 0

    if broken or bio_pct > 60:
        grade = "C"
        proba = {"A": 0.0, "B": 0.0, "C": 1.0}
    elif bio_pct > 25:
        grade = "B"
        b_score = (bio_pct - 25) / 35
        proba = {"A": 0.0, "B": round(1 - b_score * 0.3, 4), "C": round(b_score * 0.3, 4)}
    else:
        grade = "A"
        a_score = bio_pct / 25
        proba = {"A": round(1 - a_score * 0.4, 4), "B": round(a_score * 0.4, 4), "C": 0.0}

    return {"grade": grade, "probabilities": proba}


def predict_final_grade(all_vector: list, broken_shell: bool = False) -> dict:
    # pkl used here for rf_grade reference only
    _load_models()
    x        = np.array([all_vector])
    proba_rf = _rf_stage2.predict_proba(x)[0]
    rf_grade = _le.classes_[int(np.argmax(proba_rf))]

    from services.features import ALL_COLS
    feat       = dict(zip(ALL_COLS, all_vector))
    meat_ratio = feat['meat_shell_ratio']
    color_dev  = feat['flesh_color_dev']
    bio_pct    = feat['bio_coverage_pct']

    if broken_shell:
        grade = "C"
    elif meat_ratio > 25 and color_dev <= 15 and bio_pct <= 25:
        grade = "A"
    elif meat_ratio >= 20 and color_dev <= 30 and bio_pct <= 60:
        grade = "B"
    else:
        grade = "C"

    if grade == "A":
        a_conf = min(1.0, (meat_ratio - 25) / 25 * 0.3 + 0.7)
        proba_dict = {"A": round(a_conf, 4), "B": round(1 - a_conf, 4), "C": 0.0}
    elif grade == "B":
        proba_dict = {"A": 0.0, "B": 0.7, "C": 0.3}
    else:
        proba_dict = {"A": 0.0, "B": 0.0, "C": 1.0}

    return {
        "grade":                 grade,
        "rf_grade":              rf_grade,       # ← RF still used as reference
        "broken_shell_override": broken_shell,
        "probabilities":         proba_dict,
    }