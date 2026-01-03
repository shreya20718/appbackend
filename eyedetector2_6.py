# spectacle_backend.py
"""
AdvancedEyeSpectacleBackend
Backend-ready, headless converter of the "full system" logic from your GUI app.
Accepts a single BGR image (numpy array) and returns a dict with:
 - pd_mm (full interpupillary distance in mm) or None
 - pd_left_mm, pd_right_mm (distance from each pupil to nose-line) or None
 - left_center, right_center (pixel coords) if detected
 - sunglasses/frame detection confidence and flags
 - hand blocking flags
 - head_tilt_deg
 - scale_used (mm per pixel or px_per_mm equivalent)
 - stability / validity flags and messages
This module uses MediaPipe (face_mesh + hands) and OpenCV + numpy.
"""

import math
import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple, Optional, Dict, Any, List

# ---- Config ----
IRIS_REAL_MM = 11.7  # physical iris diameter used for scale estimation
MIN_VALID_PD_MM = 50
MAX_VALID_PD_MM = 75
MIN_VALID_HALF_PD_MM = 25
MAX_VALID_HALF_PD_MM = 40
ALIGNMENT_IRIS_PX_MIN = 6
ALIGNMENT_IRIS_PX_MAX = 45
HEAD_TILT_LIMIT_DEG = 6.0

# MediaPipe init (singletons; re-usable across requests)
_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
_mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Useful landmark indices (MediaPipe face mesh)
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]   # left iris (refined)
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]  # right iris (refined)
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# ---- Helpers ----
def dist(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0]-b[0], a[1]-b[1])

def is_valid_pd(pd_mm: float, half_pd: bool=False) -> bool:
    if pd_mm is None:
        return False
    if half_pd:
        return MIN_VALID_HALF_PD_MM <= pd_mm <= MAX_VALID_HALF_PD_MM
    return MIN_VALID_PD_MM <= pd_mm <= MAX_VALID_PD_MM

def landmark_to_px(landmark, w: int, h: int) -> Tuple[int,int]:
    return (int(landmark.x * w), int(landmark.y * h))

def px_from_indices(lm_list, indices, w, h) -> List[Tuple[int,int]]:
    pts = []
    for idx in indices:
        if idx < len(lm_list):
            lm = lm_list[idx]
            pts.append((int(lm.x * w), int(lm.y * h)))
    return pts

# Sunglasses/frames detection (single-frame heuristic adapted from GUI code)
def detect_sunglasses_or_frames_single(frame_bgr: np.ndarray,
                                       left_eye_px: Optional[Tuple[int,int]],
                                       right_eye_px: Optional[Tuple[int,int]],
                                       face_landmarks) -> Tuple[bool,bool,float]:
    """
    Returns (left_blocked, right_blocked, confidence)
    """
    if left_eye_px is None or right_eye_px is None or face_landmarks is None:
        return False, False, 0.0

    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def check_eye(eye_px):
        x, y = int(eye_px[0]), int(eye_px[1])
        radius = 18
        if x-radius < 0 or x+radius >= w or y-radius < 0 or y+radius >= h:
            return False, 0.0
        region = gray[y-radius:y+radius, x-radius:x+radius]
        if region.size == 0:
            return False, 0.0
        mean_int = float(np.mean(region))
        std_int = float(np.std(region))
        lap = cv2.Laplacian(region, cv2.CV_64F)
        edge_var = float(np.var(lap))

        # Heuristics
        is_dark = mean_int < 55
        is_low_contrast = std_int < 18
        is_low_texture = edge_var < 12

        dark_score = max(0, (55 - mean_int) / 55) if is_dark else 0.0
        contrast_score = max(0, (18 - std_int) / 18) if is_low_contrast else 0.0
        texture_score = max(0, (12 - edge_var) / 12) if is_low_texture else 0.0

        combined = (dark_score * 0.4 + contrast_score * 0.35 + texture_score * 0.25)
        blocked = (is_dark or is_low_contrast or is_low_texture) and combined > 0.25
        return blocked, combined

    left_blocked, left_conf = check_eye(left_eye_px)
    right_blocked, right_conf = check_eye(right_eye_px)
    return left_blocked, right_blocked, float((left_conf + right_conf) / 2.0)

# Hand covering detection (single-frame)
def check_hands_covering_eyes_single(hand_results, left_eye_px, right_eye_px, w, h) -> Tuple[bool,bool]:
    left_covered = False
    right_covered = False
    if hand_results is None or not getattr(hand_results, 'multi_hand_landmarks', None):
        return False, False

    # Convert normalized eye positions to normalized coords for easier dist calc
    def norm_eye(eye):
        return (eye[0] / w, eye[1] / h)

    left_norm = norm_eye(left_eye_px) if left_eye_px else None
    right_norm = norm_eye(right_eye_px) if right_eye_px else None

    for hand_landmarks in hand_results.multi_hand_landmarks:
        # check fingertips and palm base
        key_indices = [4, 8, 12, 16, 20, 0, 9]
        for ki in key_indices:
            lm = hand_landmarks.landmark[ki]
            if left_norm:
                d = math.hypot(lm.x - left_norm[0], lm.y - left_norm[1])
                if d < 0.09:
                    left_covered = True
            if right_norm:
                d = math.hypot(lm.x - right_norm[0], lm.y - right_norm[1])
                if d < 0.09:
                    right_covered = True
    return left_covered, right_covered

# Head tilt single-frame estimate
def compute_head_tilt_single(face_landmarks, w, h) -> float:
    try:
        left_cheek = (int(face_landmarks.landmark[LEFT_CHEEK].x * w),
                      int(face_landmarks.landmark[LEFT_CHEEK].y * h))
        right_cheek = (int(face_landmarks.landmark[RIGHT_CHEEK].x * w),
                       int(face_landmarks.landmark[RIGHT_CHEEK].y * h))
        dx = right_cheek[0] - left_cheek[0]
        dy = right_cheek[1] - left_cheek[1]
        angle = math.degrees(math.atan2(dy, dx))
        return round(angle, 2)
    except Exception:
        return 0.0

# Estimate scale mm per px using iris landmarks when available, else fallback to eye width assumption
def estimate_scale_from_face(face_landmarks, w, h) -> Tuple[Optional[float], Dict[str,Any]]:
    """
    Returns scale (mm_per_px) -- i.e. how many mm per pixel.
      - If iris landmarks available: use median iris diameter ~ IRIS_REAL_MM
      - Else fallback: use left eye outer-inner width and assume ~30 mm eye width
    Also returns diagnostics dict.
    """
    diagnostics = {}
    try:
        lm = face_landmarks.landmark
        # try iris
        left_iris_pts = px_from_indices(lm, LEFT_IRIS_IDX, w, h)
        right_iris_pts = px_from_indices(lm, RIGHT_IRIS_IDX, w, h)
        iris_diams = []
        if len(left_iris_pts) >= 3:
            # approximate iris diameter using max pairwise distance among iris points
            dd = max(dist(p1, p2) for i,p1 in enumerate(left_iris_pts) for p2 in left_iris_pts[i+1:])
            iris_diams.append(dd)
        if len(right_iris_pts) >= 3:
            dd = max(dist(p1, p2) for i,p1 in enumerate(right_iris_pts) for p2 in right_iris_pts[i+1:])
            iris_diams.append(dd)

        if iris_diams:
            iris_px = float(np.median(iris_diams))
            if iris_px > 0:
                mm_per_px = IRIS_REAL_MM / iris_px
                diagnostics["method"] = "iris"
                diagnostics["iris_px"] = iris_px
                diagnostics["mm_per_px"] = mm_per_px
                return mm_per_px, diagnostics
        # fallback: left eye outer-inner
        try:
            left_outer = landmark_to_px(lm[LEFT_EYE_OUTER], w, h)
            left_inner = landmark_to_px(lm[LEFT_EYE_INNER], w, h)
            eye_width_px = dist(left_outer, left_inner)
            if eye_width_px > 0:
                mm_per_px = 30.0 / eye_width_px  # assume eye width ~30mm
                diagnostics["method"] = "eye_width_fallback"
                diagnostics["eye_width_px"] = eye_width_px
                diagnostics["mm_per_px"] = mm_per_px
                return mm_per_px, diagnostics
        except Exception:
            pass
    except Exception:
        pass
    diagnostics["method"] = "fallback_constant"
    diagnostics["mm_per_px"] = 1.0/3.0  # px -> mm fallback (approx px_per_mm = 3 => mm_per_px = 1/3)
    return diagnostics["mm_per_px"], diagnostics

# ---- Main backend class ----
class AdvancedEyeSpectacleBackend:
    def __init__(self):
        # re-use global mediapipe instances for performance
        self.face_mesh = _mp_face_mesh
        self.hands = _mp_hands

    def process_bgr(self, frame_bgr: np.ndarray) -> Dict[str,Any]:
        """
        Take a single BGR image (OpenCV), return measurements dict.
        Non-blocking; does not open windows.
        """
        h, w = frame_bgr.shape[:2]
        result = {
            "status": "NO_FACE",
            "pd_mm": None,
            "pd_left_mm": None,
            "pd_right_mm": None,
            "left_center": None,
            "right_center": None,
            "sunglasses_detected": False,
            "sunglasses_confidence": 0.0,
            "left_hand_blocking": False,
            "right_hand_blocking": False,
            "head_tilt_deg": 0.0,
            "scale_mm_per_px": None,
            "scale_diagnostics": {},
            "warnings": []
        }

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb)
        hand_results = self.hands.process(rgb)

        if not face_results or not getattr(face_results, 'multi_face_landmarks', None):
            result["status"] = "NO_FACE"
            return result

        face_landmarks = face_results.multi_face_landmarks[0].landmark
        # get key points
        # pick central iris landmarks if available, else fallback to approximate
        left_iris_pts = px_from_indices(face_landmarks, LEFT_IRIS_IDX, w, h)
        right_iris_pts = px_from_indices(face_landmarks, RIGHT_IRIS_IDX, w, h)

        # compute centers
        left_center = tuple(np.mean(left_iris_pts, axis=0)) if left_iris_pts else None
        right_center = tuple(np.mean(right_iris_pts, axis=0)) if right_iris_pts else None

        # rounding to ints where appropriate for consistent JSON
        if left_center:
            left_center = (float(left_center[0]), float(left_center[1]))
        if right_center:
            right_center = (float(right_center[0]), float(right_center[1]))

        result["left_center"] = left_center
        result["right_center"] = right_center

        # estimate scale
        mm_per_px, diag = estimate_scale_from_face(face_results.multi_face_landmarks[0], w, h)
        result["scale_mm_per_px"] = mm_per_px
        result["scale_diagnostics"] = diag

        # check hands covering eyes
        lh, rh = False, False
        try:
            lh, rh = check_hands_covering_eyes_single(hand_results, left_center, right_center, w, h)
        except Exception:
            lh, rh = False, False
        result["left_hand_blocking"] = lh
        result["right_hand_blocking"] = rh

        # sunglasses / frame detection
        sf_left, sf_right, sf_conf = detect_sunglasses_or_frames_single(frame_bgr, left_center, right_center, face_results.multi_face_landmarks[0])
        result["sunglasses_detected"] = bool(sf_left and sf_right or sf_conf > 0.65)
        result["sunglasses_confidence"] = sf_conf

        # compute head tilt
        result["head_tilt_deg"] = compute_head_tilt_single(face_results.multi_face_landmarks[0], w, h)

        # compute PDs
        pd_mm = None
        left_pd_mm = None
        right_pd_mm = None

        # can compute left/right nose distances using nose tip and eye-line y (approach from code2)
        try:
            nose_px = landmark_to_px(face_landmarks[NOSE_TIP], w, h)
        except Exception:
            nose_px = None

        # Final pupil positions: if missing, fallback to iris approximate centers; if still missing, try some other landmarks
        def fallback_pupil(face_landmarks, primary_indices, alt_idx):
            pts = px_from_indices(face_landmarks, primary_indices, w, h)
            if pts:
                c = tuple(np.mean(pts, axis=0))
                return (float(c[0]), float(c[1]))
            # fallback to single landmark
            try:
                lm = face_landmarks[alt_idx]
                return (float(lm.x*w), float(lm.y*h))
            except Exception:
                return None

        final_left = fallback_pupil(face_landmarks, LEFT_IRIS_IDX, 469 if 469 < len(face_landmarks) else LEFT_EYE_INNER)
        final_right = fallback_pupil(face_landmarks, RIGHT_IRIS_IDX, 474 if 474 < len(face_landmarks) else RIGHT_EYE_INNER)

        # compute ipd in px if both present
        if final_left and final_right:
            ipd_px = dist(final_left, final_right)
            # mm_per_px may be zero/None if estimate failed -> guard
            if mm_per_px and mm_per_px > 0:
                pd_mm = ipd_px * mm_per_px
        # compute left/right nose distances (half-PD style) if pupil and nose present
        if final_left and nose_px and mm_per_px and mm_per_px > 0:
            left_px = dist(final_left, (nose_px[0], (final_left[1] + final_right[1]) / 2.0 if final_right else nose_px[1]))
            left_pd_mm = left_px * mm_per_px
        if final_right and nose_px and mm_per_px and mm_per_px > 0:
            right_px = dist(final_right, (nose_px[0], (final_left[1] + final_right[1]) / 2.0 if final_left else nose_px[1]))
            right_pd_mm = right_px * mm_per_px

        result["pd_mm"] = float(round(pd_mm,2)) if pd_mm else None
        result["pd_left_mm"] = float(round(left_pd_mm,2)) if left_pd_mm else None
        result["pd_right_mm"] = float(round(right_pd_mm,2)) if right_pd_mm else None

        # validity checks & warnings
        warnings = []
        if pd_mm:
            if not is_valid_pd(pd_mm):
                warnings.append("PD outside typical range")
        else:
            warnings.append("PD not computed (insufficient landmarks)")

        if (result["sunglasses_detected"] or sf_left or sf_right):
            warnings.append("Possible sunglasses/opaque frames detected - measurements may be invalid")

        if lh and rh:
            warnings.append("Hands appear to be blocking both eyes")
        elif lh:
            warnings.append("Left hand near left eye")
        elif rh:
            warnings.append("Right hand near right eye")

        # tilt check
        if abs(result["head_tilt_deg"]) > HEAD_TILT_LIMIT_DEG:
            warnings.append(f"Head tilt ({result['head_tilt_deg']}°) exceeds recommended {HEAD_TILT_LIMIT_DEG}°")

        result["warnings"] = warnings
        result["status"] = "OK"
        # include a simple confidence summary
        confidence = 0.5
        if pd_mm:
            confidence += 0.3
        if not (lh or rh) and not result["sunglasses_detected"]:
            confidence += 0.2
        result["confidence_estimate"] = min(1.0, confidence)

        return result

# Convenience helper for Flask (example)
def process_image_bytes(image_bytes: bytes) -> Dict[str,Any]:
    """
    Accept raw bytes (e.g. uploaded file content), decode to BGR CV image and process.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"status":"INVALID_IMAGE"}
    backend = AdvancedEyeSpectacleBackend()
    return backend.process_bgr(img)

# If you want to re-use the backend without reinitializing mediapipe each request,
# instantiate AdvancedEyeSpectacleBackend once at module import and reuse.

# Example Flask endpoint (paste into your Flask app)
FLASK_EXAMPLE = """
from flask import Flask, request, jsonify
from spectacle_backend import AdvancedEyeSpectacleBackend
app = Flask(__name__)
backend = AdvancedEyeSpectacleBackend()

@app.route('/measure_pd', methods=['POST'])
def measure_pd():
    # expects form-data file field named 'image' (png/jpg) or raw bytes
    if 'image' in request.files:
        file = request.files['image']
        img_bytes = file.read()
        from spectacle_backend import process_image_bytes
        res = process_image_bytes(img_bytes)
        return jsonify(res)
    else:
        return jsonify({"status":"NO_IMAGE"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""

