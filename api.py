# api.py
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

import io
import cv2
import numpy as np
from PIL import Image, ImageOps

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional

from eyedetector2_6 import AdvancedEyeSpectacleBackend, HEAD_TILT_LIMIT_DEG
from frame import process_frame as progressive_process   # ✅ NEW

app = FastAPI(title="Spectacle Eye API", version="1.0")
backend = AdvancedEyeSpectacleBackend()

MIN_DIST_CM = 30.0
MAX_DIST_CM = 60.0


def _pt_to_list(pt) -> Optional[list]:
    try:
        if pt is None:
            return None
        if isinstance(pt, (tuple, list)) and len(pt) >= 2:
            return [int(pt[0]), int(pt[1])]
        return None
    except Exception:
        return None


def _reject(message: str, code: str, extra: Optional[Dict[str, Any]] = None):
    payload = {
        "ok": False,
        "accepted": False,
        "code": code,
        "message": message,
    }
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=200, content=payload)


@app.get("/ping")
def root():
    return {"ok": True, "message": "Spectacle Eye API is running"}


def _decode_image_with_exif_fix(raw_bytes: bytes) -> np.ndarray:
    pil_img = Image.open(io.BytesIO(raw_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):

    # ---------------- Image Decode ----------------
    try:
        raw_bytes = await image.read()
        frame_bgr = _decode_image_with_exif_fix(raw_bytes)

        if frame_bgr is None or frame_bgr.size == 0:
            return _reject("Invalid image file.", code="INVALID_IMAGE")

    except Exception as e:
        return _reject(f"Image decoding failed: {str(e)}", code="DECODE_FAILED")

    img_h, img_w = frame_bgr.shape[:2]

    # ---------------- MAIN BACKEND ----------------
    result = backend.process_frame(frame_bgr)

    # ---------------- PROGRESSIVE ENGINE (NEW) ----------------
    prog_data = progressive_process(frame_bgr)

    prog_left_pupil = None
    prog_right_pupil = None

    if prog_data.get("face_detected"):
        lp = prog_data.get("pupils", {}).get("left")
        rp = prog_data.get("pupils", {}).get("right")

        if lp:
            prog_left_pupil = [int(lp["x"]), int(lp["y"])]
        if rp:
            prog_right_pupil = [int(rp["x"]), int(rp["y"])]

    # ----------------------------------------------------------

    auto_left_pupil = _pt_to_list(result.get("auto_left_pupil"))
    auto_right_pupil = _pt_to_list(result.get("auto_right_pupil"))
    scale_mm_per_px = result.get("scale_mm_per_px")

    base_extra = {
        "img_w": int(img_w),
        "img_h": int(img_h),
        "auto_left_pupil": auto_left_pupil,
        "auto_right_pupil": auto_right_pupil,
        "progressive_left_pupil": prog_left_pupil,       # ✅ ADDED
        "progressive_right_pupil": prog_right_pupil,     # ✅ ADDED
        "scale_mm_per_px": scale_mm_per_px,
        "warnings": result.get("warnings", []),
    }

    if not result.get("ok", False):
        return _reject("Face not detected.", code="NO_FACE", extra=base_extra)

    # ---------------- Extract fields ----------------
    distance_cm = result.get("distance_cm")
    head_tilt = float(result.get("head_tilt_deg") or 0.0)

    left_eye_open = bool(result.get("left_eye_open"))
    right_eye_open = bool(result.get("right_eye_open"))

    left_hand = bool(result.get("left_hand_blocking"))
    right_hand = bool(result.get("right_hand_blocking"))

    left_frame = bool(result.get("left_frame_blocking"))
    right_frame = bool(result.get("right_frame_blocking"))
    frame_conf = float(result.get("frame_detection_confidence") or 0.0)

    pd_total = result.get("pd_mm")
    pd_left = result.get("pd_left_mm")
    pd_right = result.get("pd_right_mm")

    # ---------------- Validations ----------------
    if abs(head_tilt) > float(HEAD_TILT_LIMIT_DEG):
        extra = dict(base_extra)
        extra.update({"head_tilt_deg": head_tilt})
        return _reject("Head tilt too high.", code="HEAD_TILT_HIGH", extra=extra)

    if distance_cm is None:
        extra = dict(base_extra)
        return _reject("Distance not estimated.", code="DISTANCE_NOT_AVAILABLE", extra=extra)

    if not (MIN_DIST_CM <= float(distance_cm) <= MAX_DIST_CM):
        extra = dict(base_extra)
        extra.update({"distance_cm": float(distance_cm)})
        return _reject("Invalid distance.", code="BAD_DISTANCE", extra=extra)

    if left_hand or right_hand:
        extra = dict(base_extra)
        return _reject("Hand detected near eye.", code="HAND_BLOCKING", extra=extra)

    if not left_eye_open or not right_eye_open:
        extra = dict(base_extra)
        return _reject("Eyes closed.", code="EYES_CLOSED", extra=extra)

    sunglasses_detected = bool(left_frame or right_frame)
    pd_ok = (pd_total is not None and pd_left is not None and pd_right is not None)

    # ---------------- FINAL RESPONSE ----------------
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "accepted": True,

            "pd_ok": pd_ok,
            "pd_total_mm": float(pd_total) if pd_total else None,
            "pd_left_mm": float(pd_left) if pd_left else None,
            "pd_right_mm": float(pd_right) if pd_right else None,

            "img_w": int(img_w),
            "img_h": int(img_h),

            "auto_left_pupil": auto_left_pupil,
            "auto_right_pupil": auto_right_pupil,

            # ✅ Progressive pupils
            "progressive_left_pupil": prog_left_pupil,
            "progressive_right_pupil": prog_right_pupil,

            "scale_mm_per_px": scale_mm_per_px,
            "sunglasses_or_frames_detected": sunglasses_detected,
            "frame_detection_confidence": float(frame_conf),

            "distance_cm": float(distance_cm),
            "head_tilt_deg": float(head_tilt),
            "left_eye_open": left_eye_open,
            "right_eye_open": right_eye_open,
            "left_hand_blocking": left_hand,
            "right_hand_blocking": right_hand,

            "warnings": result.get("warnings", []),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=10000, reload=False)

