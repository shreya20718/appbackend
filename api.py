# Spectacle Eye API
# ===============================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

import io
import uuid
import cv2
import numpy as np
from PIL import Image, ImageOps

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from eyedetector2_6 import SpectacleEyeBackend, HEAD_TILT_LIMIT_DEG
from frame import process_frame as progressive_process


# ===============================
# App Init
# ===============================

app = FastAPI(title="Spectacle Eye API", version="1.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

backend = SpectacleEyeBackend()

MIN_DIST_CM = 30.0
MAX_DIST_CM = 60.0

PDF_DIR = os.path.join(os.getcwd(), "pdf_storage")
os.makedirs(PDF_DIR, exist_ok=True)



# ===============================
# Helper Functions
# ===============================

def _pt_to_list(pt) -> Optional[list]:
    if pt is None:
        return None
    if isinstance(pt, (tuple, list)) and len(pt) >= 2:
        return [float(pt[0]), float(pt[1])]
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


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    pil_img = Image.open(io.BytesIO(raw_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_pdf_on_server(data: dict):

    try:

        filename = f"report_{uuid.uuid4().hex}.pdf"
        filepath = os.path.join(PDF_DIR, filename)

        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Spectacle Eye Report", styles["Title"]))
        elements.append(Spacer(1, 20))

        for key, value in data.items():
            elements.append(Paragraph(f"{key}: {value}", styles["Normal"]))
            elements.append(Spacer(1, 10))

        doc.build(elements)

        print("PDF saved at:", filepath)

    except Exception as e:
        print("PDF generation error:", str(e))


# ===============================
# Routes
# ===============================

from fastapi.responses import FileResponse


@app.get("/reports/{filename}")
def get_report(filename: str):
    filepath = os.path.join(PDF_DIR, filename)

    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/pdf")

    return {"error": "File not found"}


@app.get("/ping")
def ping():
    return {"ok": True, "message": "Spectacle Eye API is running"}


from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
import json

@app.post("/analyze")
async def analyze_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    manual_left: Optional[str] = Form(None),
    manual_right: Optional[str] = Form(None)
):
    # ---- Reset State for Static Analysis ----
    backend.reset_history()

    # ---- Parse manual points if provided ----
    m_left = None
    m_right = None
    try:
        if manual_left:
            m_left = json.loads(manual_left)
        if manual_right:
            m_right = json.loads(manual_right)
    except Exception as e:
        print(f"Warning: could not parse manual points: {e}")

    # ---- Decode Image ----
    try:
        raw_bytes = await image.read()


        frame_bgr = _decode_image(raw_bytes)

        if frame_bgr is None or frame_bgr.size == 0:
            return _reject("Invalid image file.", "INVALID_IMAGE")

    except Exception as e:
        return _reject(f"Image decoding failed: {str(e)}", "DECODE_FAILED")

    img_h, img_w = frame_bgr.shape[:2]

    # ---- Main Detection ----
    result = backend.process_frame(frame_bgr, manual_left_pupil=m_left, manual_right_pupil=m_right)
    prog_data = {} # Remove progressive process usage as requested

    # ---- DEBUG: Save Detect Image ----
    try:
        debug_frame = frame_bgr.copy()
        l_pupil = result.get("auto_left_pupil")
        r_pupil = result.get("auto_right_pupil")
        if l_pupil:
            cv2.circle(debug_frame, (int(l_pupil[0]), int(l_pupil[1])), 10, (0, 255, 0), -1)
        if r_pupil:
            cv2.circle(debug_frame, (int(r_pupil[0]), int(r_pupil[1])), 10, (0, 255, 0), -1)
        
        debug_path = os.path.join(PDF_DIR, "debug_last_capture.jpg")
        cv2.imwrite(debug_path, debug_frame)
        print(f"Debug image saved to: {debug_path}")
    except Exception as e:
        print(f"Failed to save debug image: {e}")

    if not result.get("ok", False):

        return _reject("Face not detected.", "NO_FACE")

    # ---- Progressive Pupils ----
    prog_left_pupil = None
    prog_right_pupil = None

    if prog_data.get("face_detected"):

        lp = prog_data.get("pupils", {}).get("left")
        rp = prog_data.get("pupils", {}).get("right")

        if lp:
            prog_left_pupil = [int(lp["x"]), int(lp["y"])]

        if rp:
            prog_right_pupil = [int(rp["x"]), int(rp["y"])]

    auto_left_pupil = _pt_to_list(result.get("auto_left_pupil"))
    auto_right_pupil = _pt_to_list(result.get("auto_right_pupil"))

    # ---- Extract Values ----
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

    # ---- Validations ----

    if distance_cm is None:
        return _reject("Distance not estimated.", "DISTANCE_NOT_AVAILABLE")

    d_val = float(distance_cm)
    if d_val < MIN_DIST_CM:
        return _reject("Move far", "TOO_CLOSE")
    if d_val > MAX_DIST_CM:
        return _reject("Move closer", "TOO_FAR")

    if abs(head_tilt) > float(HEAD_TILT_LIMIT_DEG):
        return _reject("Head tilt too high.", "HEAD_TILT_HIGH")

    if left_hand or right_hand:
        return _reject("Hand detected near eye.", "HAND_BLOCKING")

    if not left_eye_open or not right_eye_open:
        return _reject("Eyes closed.", "EYES_CLOSED")

    sunglasses_detected = bool(left_frame or right_frame)

   

    
    # Symmetry Check
    pd_ok = pd_total is not None

    # ---- Final Response ----
    response_payload = {

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

        "distance_cm": float(distance_cm),
        "head_tilt_deg": float(head_tilt),

        "left_eye_open": left_eye_open,
        "right_eye_open": right_eye_open,

        "sunglasses_or_frames_detected": sunglasses_detected,
        "frame_detection_confidence": frame_conf,

        "scale_mm_per_px": result.get("scale_mm_per_px"),
        "nose_center_px": result.get("nose_center_px"),

        "warnings": result.get("warnings", []),
    }


    # Save PDF only if accepted
    background_tasks.add_task(save_pdf_on_server, response_payload)

    return JSONResponse(status_code=200, content=response_payload)


# ===============================
# Local Run (Only for Testing)
# ===============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
