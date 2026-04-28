from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2
import io

from models.maskrcnn import run_inference
from models.classifier import predict_initial_grade, predict_final_grade
from services.features import (
    extract_side_features, average_side_features,
    extract_meat_features, build_stage1_vector,
    build_all_vector, is_broken_shell,
)
from services.supabase import upload_image, get_session, update_session

router = APIRouter()


def _read_bgr(upload_file) -> np.ndarray:
    """Decode an UploadFile to a BGR numpy array (cv2 format)."""
    data = upload_file.file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {upload_file.filename}")
    return img


@router.post("/initial/{session_id}")
async def initial_grade(session_id: str,
                         shell_a: UploadFile = File(...),
                         shell_b: UploadFile = File(...)):
    img_a = _read_bgr(shell_a)
    img_b = _read_bgr(shell_b)

    result_a = run_inference(img_a)
    result_b = run_inference(img_b)

    feat_a   = extract_side_features(result_a)
    feat_b   = extract_side_features(result_b)
    combined = average_side_features(feat_a, feat_b)
    broken   = is_broken_shell(combined)

    vec_s1   = build_stage1_vector(combined)
    result   = predict_initial_grade(vec_s1)

    shell_a.file.seek(0)
    shell_b.file.seek(0)
    path_a = upload_image(session_id, "shell_a", shell_a.file.read())
    path_b = upload_image(session_id, "shell_b", shell_b.file.read())

    update_session(session_id, {
        "stage":            2,
        "shell_a_path":     path_a,
        "shell_b_path":     path_b,
        "initial_grade":    result["grade"],
        "initial_features": combined,
    })

    return {
        "grade":         result["grade"],
        "probabilities": result["probabilities"],
        "broken_shell":  broken,
        "features":      combined,
        "shell_a":       result_a["shell_b64"],    # ← new
        "shell_b":       result_b["shell_b64"],    # ← new
        "bio_a":         result_a["bio_b64"],      # ← new
        "bio_b":         result_b["bio_b64"],      # ← new
        "overlay_a":     result_a["overlay_b64"],
        "overlay_b":     result_b["overlay_b64"],
    }


@router.post("/final/{session_id}")
async def final_grade(session_id: str,
                       meat: UploadFile = File(...)):
    session = get_session(session_id)
    if not session or not session.get("initial_features"):
        raise HTTPException(400, "Complete initial grading first")

    img_meat    = _read_bgr(meat)
    result_meat = run_inference(img_meat)

    combined    = session["initial_features"]
    broken      = is_broken_shell(combined)

    meat_feats  = extract_meat_features(result_meat, combined["shell_mm2"])
    vec_all     = build_all_vector(combined, meat_feats)

    result      = predict_final_grade(vec_all, broken_shell=broken)

    meat.file.seek(0)
    path_meat = upload_image(session_id, "meat", meat.file.read())

    update_session(session_id, {
        "stage":         4,
        "meat_path":     path_meat,
        "final_grade":   result["grade"],
        "final_features": {**combined, **meat_feats},
    })

    return {
        "session_id":            session_id,
        "grade":                 result["grade"],
        "rf_grade":              result["rf_grade"],
        "broken_shell_override": result["broken_shell_override"],
        "probabilities":         result["probabilities"],
        "features":              {**combined, **meat_feats},
        "overlay_meat":          result_meat["overlay_b64"],
    }