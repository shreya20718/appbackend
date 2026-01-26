import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

# =====================================================
# Optical Constants
# =====================================================
IRIS_REAL_MM = 11.7
SMOOTH_FRAMES = 8

DEFAULT_Y_OFFSET_MM = 14.0
DEFAULT_X_OFFSET_MM = 6.0

# =====================================================
# MediaPipe Setup (ONE TIME INIT)
# =====================================================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================================================
# State (Persistent Across Frames)
# =====================================================
left_iris_hist = deque(maxlen=SMOOTH_FRAMES)
right_iris_hist = deque(maxlen=SMOOTH_FRAMES)

# Stored in millimeters
offsets_mm = {
    "left_top":    [0.0, -5.0],
    "left_bottom": [DEFAULT_X_OFFSET_MM,  DEFAULT_Y_OFFSET_MM],
    "right_top":   [0.0, -5.0],
    "right_bottom":[-DEFAULT_X_OFFSET_MM, DEFAULT_Y_OFFSET_MM],
}

# =====================================================
# Utilities
# =====================================================
def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


# =====================================================
# MAIN FUNCTION YOU CALL FROM api.py
# =====================================================
def process_frame(frame_bgr):
    """
    Input  : BGR image (numpy array)
    Output : dict with all progressive lens data
    """

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    output = {
        "face_detected": False,
        "scale_px_per_mm": None,
        "pupils": {},
        "points_px": {},
        "offsets_mm": offsets_mm
    }

    if not res.multi_face_landmarks:
        return output

    lm = res.multi_face_landmarks[0].landmark

    # =================================================
    # Pupil Centers
    # =================================================
    pupils = {
        "left":  (lm[468].x * w, lm[468].y * h),
        "right": (lm[473].x * w, lm[473].y * h)
    }

    # =================================================
    # Iris Size for Scale
    # =================================================
    left_iris = dist(
        (lm[469].x * w, lm[469].y * h),
        (lm[471].x * w, lm[471].y * h)
    )
    right_iris = dist(
        (lm[474].x * w, lm[474].y * h),
        (lm[476].x * w, lm[476].y * h)
    )

    if left_iris > 0:
        left_iris_hist.append(left_iris)
    if right_iris > 0:
        right_iris_hist.append(right_iris)

    iris_vals = []
    if left_iris_hist:
        iris_vals.append(np.median(left_iris_hist))
    if right_iris_hist:
        iris_vals.append(np.median(right_iris_hist))

    if not iris_vals:
        return output

    px_per_mm = (sum(iris_vals) / len(iris_vals)) / IRIS_REAL_MM

    # =================================================
    # Convert Offsets → Pixel Coordinates
    # =================================================
    points_px = {}

    for eye in ["left", "right"]:
        px, py = pupils[eye]

        for pos in ["top", "bottom"]:
            key = f"{eye}_{pos}"
            dx_mm, dy_mm = offsets_mm[key]

            cx = px + dx_mm * px_per_mm
            cy = py + dy_mm * px_per_mm

            points_px[key] = {
                "x": float(cx),
                "y": float(cy),
                "dx_mm": dx_mm,
                "dy_mm": dy_mm
            }

    # =================================================
    # Final Structured Output
    # =================================================
    output["face_detected"] = True
    output["scale_px_per_mm"] = float(px_per_mm)
    output["pupils"] = {
        "left": {"x": float(pupils["left"][0]), "y": float(pupils["left"][1])},
        "right": {"x": float(pupils["right"][0]), "y": float(pupils["right"][1])}
    }
    output["points_px"] = points_px

    return output
