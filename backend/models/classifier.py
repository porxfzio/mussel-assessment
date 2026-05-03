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
    """
    Stage 1 — Initial grade using rf_stage1_shell.pkl.
    The RF predicts grade based on the 13 external shell features.
    If a broken shell is detected (side_meat_pct > 0), grade is
    overridden to C regardless of RF output.
    """
    _load_models()

    from services.features import STAGE1_COLS
    feat   = dict(zip(STAGE1_COLS, stage1_vector))
    broken = feat['side_meat_pct'] > 0

    x        = np.array([stage1_vector])
    proba_rf = _rf_stage1.predict_proba(x)[0]
    classes  = _le.classes_                          # ['A', 'B', 'C']
    rf_grade = classes[int(np.argmax(proba_rf))]

    # Build probability dict aligned to label encoder classes
    proba_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, proba_rf)}

    # Broken shell override — exposed meat on exterior = Grade C
    if broken:
        grade      = "C"
        proba_dict = {"A": 0.0, "B": 0.0, "C": 1.0}
    else:
        grade = rf_grade

    return {
        "grade":       grade,
        "probabilities": proba_dict,
    }


def predict_final_grade(all_vector: list, broken_shell: bool = False) -> dict:
    """
    Stage 2 — Final grade using rf_stage2_final.pkl.
    The RF predicts grade based on the full 17-feature vector.
    If a broken shell was detected in Stage 1, grade is overridden to C.
    """
    _load_models()

    x        = np.array([all_vector])
    proba_rf = _rf_stage2.predict_proba(x)[0]
    classes  = _le.classes_                          # ['A', 'B', 'C']
    rf_grade = classes[int(np.argmax(proba_rf))]

    # Build probability dict aligned to label encoder classes
    proba_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, proba_rf)}

    # Broken shell override — always Grade C
    if broken_shell:
        grade      = "C"
        proba_dict = {"A": 0.0, "B": 0.0, "C": 1.0}
    else:
        grade = rf_grade

    return {
        "grade":                 grade,
        "rf_grade":              rf_grade,        # RF prediction before override
        "broken_shell_override": broken_shell,
        "probabilities":         proba_dict,
    }