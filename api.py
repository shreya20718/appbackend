# import io
# import os
# import base64
# import traceback
# import cv2
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image

# from eyedetector2_6 import AdvancedEyeSpectacleBackend
# from backend import spectacle_system

# # ---------------- APP ----------------
# app = Flask(__name__)
# CORS(app)
# app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB

# # ---------------- LOAD BACKENDS ONCE ----------------
# pd_backend = AdvancedEyeSpectacleBackend()

# # ---------------- HELPERS ----------------
# def decode_b64_image(b64: str):
#     raw = base64.b64decode(b64, validate=True)
#     img = Image.open(io.BytesIO(raw)).convert("RGB")
#     return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# # ---------------- ROUTES ----------------
# @app.route("/ping", methods=["GET"])
# def ping():
#     return jsonify({"status": "ok"})

# @app.route("/process", methods=["POST"])
# def process():
#     try:
#         data = request.get_json(silent=True)
#         if not data or "image_b64" not in data:
#             return jsonify({"status": "ERROR", "error": "image_b64_required"}), 400

#         # ---- Decode image ----
#         frame = decode_b64_image(data["image_b64"])
#         h, w = frame.shape[:2]

#         # ---- BACKEND PROCESSING ----
#         pd_result = pd_backend.process_bgr(frame)
#         frame_result = spectacle_system.process_frame(frame)

#         frame_detected = bool(frame_result.get("detected", False))

#         # ---- STRICT RESPONSE (NO LOGIC) ----
#         response = {
#             "status": pd_result.get("status", "ERROR"),

#             "image": {
#                 "width": w,
#                 "height": h
#             },

#             "pd": {
#                 "total_mm": pd_result.get("pd_mm"),
#                 "left_mm": pd_result.get("pd_left_mm"),
#                 "right_mm": pd_result.get("pd_right_mm"),
#                 "scale_mm_per_px": pd_result.get("scale_mm_per_px")
#             },

#             "eyes": {
#                 "left_center_px": pd_result.get("left_center"),
#                 "right_center_px": pd_result.get("right_center")
#             },

#             "occlusion": {
#                 "sunglasses_detected": pd_result.get("sunglasses_detected"),
#                 "sunglasses_confidence": pd_result.get("sunglasses_confidence"),
#                 "left_hand_blocking": pd_result.get("left_hand_blocking"),
#                 "right_hand_blocking": pd_result.get("right_hand_blocking")
#             },

#             "pose": {
#                 "head_tilt_deg": pd_result.get("head_tilt_deg")
#             },

#             "frame": {
#                 "detected": frame_detected,
#                 "A_mm": frame_result.get("A_mm") if frame_detected else None,
#                 "B_mm": frame_result.get("B_mm") if frame_detected else None,
#                 "DBL_mm": frame_result.get("DBL_mm") if frame_detected else None
#             },

#             "diagnostics": {
#                 "warnings": pd_result.get("warnings", []),
#                 "confidence_estimate": pd_result.get("confidence_estimate"),
#                 "scale_diagnostics": pd_result.get("scale_diagnostics", {})
#             }
#         }

#         return jsonify(response)

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({
#             "status": "ERROR",
#             "error": str(e)
#         }), 500

# # ---------------- RUN ----------------
# if __name__ == "__main__":
#     print("🔥 STRONG backend-driven API running")

#     # warmup
#     try:
#         dummy = np.zeros((240, 320, 3), dtype=np.uint8)
#         pd_backend.process_bgr(dummy)
#         spectacle_system.process_frame(dummy)
#     except Exception:
#         pass

#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port, debug=False)
# myAPI.py

# api.py
"""
OPTIFOCUS API
-------------
Single entry-point API that:
- Accepts a base64-encoded face image
- Computes PD, eye centers, occlusions, and head tilt
- Computes spectacle frame dimensions (A, B, DBL)
- Returns a strict, frontend-safe JSON response

IMPORTANT:
- PD logic lives in AdvancedEyeSpectacleBackend (DO NOT modify here)
- Frame logic lives in spectacle_system (DO NOT modify here)
"""

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

# ---------------- APP ----------------
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB

# ---------------- INIT BACKENDS ----------------
pd_backend = AdvancedEyeSpectacleBackend()

# ---------------- HELPERS ----------------
def decode_b64_image(b64: str):
    raw = base64.b64decode(b64, validate=True)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def r1(v):
    """Round to 1 decimal if value exists"""
    return round(float(v), 1) if v is not None else None

# ---------------- ROUTES ----------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json(silent=True)
        if not data or "image_b64" not in data:
            return jsonify({"status": "ERROR", "error": "image_b64_required"}), 400

        frame = decode_b64_image(data["image_b64"])
        h, w = frame.shape[:2]

        pd_result = pd_backend.process_bgr(frame)
        frame_result = spectacle_system.process_frame(frame)

        frame_detected = bool(frame_result.get("detected", False))

        response = {
            "status": pd_result.get("status", "ERROR"),

            "image": {
                "width": w,
                "height": h
            },

            # ---------------- PD (1 decimal enforced) ----------------
            "pd": {
                "total_mm": r1(pd_result.get("pd_mm")),
                "left_mm": r1(pd_result.get("pd_left_mm")),
                "right_mm": r1(pd_result.get("pd_right_mm")),
                "scale_mm_per_px": r1(pd_result.get("scale_mm_per_px"))
            },

            "eyes": {
                "left_center_px": pd_result.get("left_center"),
                "right_center_px": pd_result.get("right_center"),
                "valid": (
                    pd_result.get("pd_mm") is not None or
                    (
                        pd_result.get("left_center") is not None and
                        pd_result.get("right_center") is not None
                    )
                )
            },

            "occlusion": {
                "sunglasses_detected": pd_result.get("sunglasses_detected"),
                "sunglasses_confidence": r1(pd_result.get("sunglasses_confidence")),
                "left_hand_blocking": pd_result.get("left_hand_blocking"),
                "right_hand_blocking": pd_result.get("right_hand_blocking")
            },

            "pose": {
                "head_tilt_deg": r1(pd_result.get("head_tilt_deg"))
            },

            # ---------------- FRAME (1 decimal enforced) ----------------
            "frame": {
                "detected": frame_detected,
                "A_mm": r1(frame_result.get("A_mm")) if frame_detected else None,
                "B_mm": r1(frame_result.get("B_mm")) if frame_detected else None,
                "DBL_mm": r1(frame_result.get("DBL_mm")) if frame_detected else None
            },

            "diagnostics": {
                "warnings": pd_result.get("warnings", []),
                "confidence_estimate": r1(pd_result.get("confidence_estimate")),
                "scale_diagnostics": pd_result.get("scale_diagnostics", {})
            }
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "ERROR", "error": str(e)}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🔥 OPTIFOCUS API running")

    try:
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        pd_backend.process_bgr(dummy)
        spectacle_system.process_frame(dummy)
    except Exception:
        pass

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
