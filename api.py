import io
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
app = Flask(__name__)
CORS(app)

pd_backend = AdvancedEyeSpectacleBackend()

# ---------------- HELPERS ----------------
def b64_to_cv2(b64):
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

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

        # ---- PD DETECTION (REAL) ----
        pd_result = pd_backend.process_bgr(frame)

        # ---- FRAME DETECTION (FLAG ONLY) ----
        frame_result = spectacle_system.process_frame(frame)
        frame_detected = bool(frame_result.get("detected", False))

        # ---- BASE RESPONSE (ALWAYS) ----
        response = {
            "status": "OK",

            # PD values
            "pd_left_mm": pd_result.get("pd_left_mm"),
            "pd_right_mm": pd_result.get("pd_right_mm"),
            "pd_total_mm": pd_result.get("pd_mm"),

            "left_eye_center_px": pd_result.get("left_center"),
            "right_eye_center_px": pd_result.get("right_center"),

            # Meta
            "frame_detected": frame_detected,
            "image_width": w,
            "image_height": h,
            "warnings": pd_result.get("warnings"),
        }

        # ---- FRAME VALUES ONLY IF FRAME IS DETECTED ----
        if frame_detected:
            response.update({
                # Static optical values (TEMP / FAKE)
                "A_mm": 45.0,
                "B_mm": 28.0,
                "DBL_mm": 16.0,
            })

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🔥 API running (REAL PD + CONDITIONAL FRAME VALUES)")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    pd_backend.process_bgr(dummy)
    spectacle_system.process_frame(dummy)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
