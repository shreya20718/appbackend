import io
import os
import base64
import traceback
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from eyedetector2_6 import AdvancedEyeSpectacleBackend
from backend import spectacle_system

# ---------------- APP SETUP ----------------
app = Flask(_name_)
CORS(app)

# 🔥 HARD LIMIT — prevents Render worker kill
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB

# 🔥 Load models ONCE (global)
pd_backend = AdvancedEyeSpectacleBackend()

# ---------------- HELPERS ----------------
def b64_to_cv2(b64):
    try:
        raw = base64.b64decode(b64, validate=True)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

# ---------------- ROUTES ----------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json(silent=True)
        if not data or "image_b64" not in data:
            return jsonify({"error": "image_b64_required"}), 400

        # ---- Decode image ----
        frame = b64_to_cv2(data["image_b64"])
        h, w = frame.shape[:2]

        # ---- PD DETECTION ----
        pd_result = pd_backend.process_bgr(frame)

        # ---- FRAME DETECTION ----
        frame_result = spectacle_system.process_frame(frame)
        frame_detected = bool(frame_result.get("detected", False))

        # ---- BASE RESPONSE ----
        response = {
            "status": "OK",

            "pd_left_mm": pd_result.get("pd_left_mm"),
            "pd_right_mm": pd_result.get("pd_right_mm"),
            "pd_total_mm": pd_result.get("pd_mm"),

            "left_eye_center_px": pd_result.get("left_center"),
            "right_eye_center_px": pd_result.get("right_center"),

            "frame_detected": frame_detected,
            "image_width": w,
            "image_height": h,
            "warnings": pd_result.get("warnings"),
        }

        # ---- FRAME VALUES ONLY IF DETECTED ----
        if frame_detected:
            response.update({
                "A_mm": 45.0,
                "B_mm": 28.0,
                "DBL_mm": 16.0,
            })

        return jsonify(response)

    except Exception as e:
        print("PROCESS ERROR:", e, flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- RUN (RENDER SAFE) ----------------
if _name_ == "_main_":
    print("🔥 API running (Render-safe, no 502)")

    # 🔥 SAFE warm-up (small dummy)
    try:
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        pd_backend.process_bgr(dummy)
        spectacle_system.process_frame(dummy)
    except Exception as e:
        print("Warmup failed:", e)

    # 🔥 REQUIRED for Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)