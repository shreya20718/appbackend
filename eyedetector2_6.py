import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

# ---------------- Config ----------------
IRIS_REAL_MM = 11.7

HISTORY_LEN = 20
PUPIL_HISTORY_LEN = 10

HEAD_TILT_LIMIT_DEG = 8.0

# PD validation ranges
MIN_VALID_PD_MM = 50
MAX_VALID_PD_MM = 75
MIN_VALID_HALF_PD_MM = 25
MAX_VALID_HALF_PD_MM = 40

# Distance estimation config (must be calibrated for real accuracy)
FOCAL_LENGTH_PX = 850  # device-specific constant (approx)


# ---------------- MediaPipe init ----------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


# ---------------- Utils ----------------
def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) if (a and b) else 0.0


def filter_outliers(values, threshold=2.0):
    if len(values) < 3:
        return values
    median = np.median(values)
    mad = np.median([abs(v - median) for v in values])
    if mad == 0:
        return values
    filtered = [v for v in values if abs(v - median) / mad < threshold]
    return filtered if filtered else values


def is_valid_pd(pd_mm: float, half_pd=False) -> bool:
    if pd_mm is None:
        return False
    if half_pd:
        return MIN_VALID_HALF_PD_MM <= pd_mm <= MAX_VALID_HALF_PD_MM
    return MIN_VALID_PD_MM <= pd_mm <= MAX_VALID_PD_MM


def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
    if len(pts) != 6:
        return 0.0
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def weighted_average_position(history):
    """
    ✅ FIXED: your earlier version had a bug (variable shadowing).
    """
    if not history:
        return None
    weights = np.linspace(0.5, 1.0, len(history))
    weights = weights / weights.sum()

    x_avg = sum(pt[0] * wt for pt, wt in zip(history, weights))
    y_avg = sum(pt[1] * wt for pt, wt in zip(history, weights))
    return (int(round(x_avg)), int(round(y_avg)))


def calculate_head_tilt_deg(face_landmarks, w, h, head_tilt_history: deque):
    """
    ✅ Correct head tilt:
    - 0° = straight
    - + = tilt right, - = tilt left (depending camera)
    """
    try:
        forehead = np.array([
            face_landmarks.landmark[10].x * w,
            face_landmarks.landmark[10].y * h
        ], dtype=np.float32)

        chin = np.array([
            face_landmarks.landmark[152].x * w,
            face_landmarks.landmark[152].y * h
        ], dtype=np.float32)

        v = chin - forehead
        vx, vy = float(v[0]), float(v[1])

        # angle between v and vertical axis
        # atan2(x, y) gives sideways deviation from vertical
        signed_angle = math.degrees(math.atan2(vx, vy))
        tilt = abs(signed_angle)

        # smooth it
        head_tilt_history.append(tilt)
        smoothed = float(sum(head_tilt_history) / len(head_tilt_history))
        return smoothed, float(signed_angle)

    except Exception:
        return 0.0, 0.0


# ---------------- Hand Blocking ----------------
def detect_hand_near_eye(hand_landmarks, eye_region, threshold=0.10):
    if not hand_landmarks or not eye_region:
        return False

    eye_x, eye_y, _ = eye_region
    key_points = [4, 8, 12, 16, 20, 0, 9]

    detection_count = 0
    required_detections = 2

    for idx in key_points:
        hand_x = hand_landmarks.landmark[idx].x
        hand_y = hand_landmarks.landmark[idx].y
        d = math.sqrt((hand_x - eye_x) ** 2 + (hand_y - eye_y) ** 2)
        if d < threshold:
            detection_count += 1
            if detection_count >= required_detections:
                return True
    return False


def check_hands_covering_eyes(hand_results, left_eye_pos, right_eye_pos, w, h):
    left_covered = False
    right_covered = False

    if not hand_results.multi_hand_landmarks:
        return False, False

    left_eye_region = None
    right_eye_region = None

    if left_eye_pos:
        left_eye_region = (left_eye_pos[0] / w, left_eye_pos[1] / h, 0.08)
    if right_eye_pos:
        right_eye_region = (right_eye_pos[0] / w, right_eye_pos[1] / h, 0.08)

    for hand_landmarks in hand_results.multi_hand_landmarks:
        hand_center_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)

        if left_eye_region:
            if detect_hand_near_eye(hand_landmarks, left_eye_region, threshold=0.09):
                if hand_center_x <= 0.65:
                    left_covered = True

        if right_eye_region:
            if detect_hand_near_eye(hand_landmarks, right_eye_region, threshold=0.09):
                if hand_center_x >= 0.35:
                    right_covered = True

    return left_covered, right_covered


# ---------------- Sunglasses / Frame Blocking ----------------
def detect_sunglasses_or_frames(frame, left_eye_pos, right_eye_pos, face_landmarks, w, h):
    if not left_eye_pos or not right_eye_pos:
        return False, False, 0.0

    left_blocked = False
    right_blocked = False
    confidence = 0.0

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def check_eye_region_opacity(eye_pos, gray_img):
            x, y = int(eye_pos[0]), int(eye_pos[1])
            radius = 18

            if x - radius < 0 or x + radius >= gray_img.shape[1] or y - radius < 0 or y + radius >= gray_img.shape[0]:
                return False, 0.0

            eye_region = gray_img[y - radius:y + radius, x - radius:x + radius]
            if eye_region.size == 0:
                return False, 0.0

            mean_intensity = np.mean(eye_region)
            std_intensity = np.std(eye_region)

            laplacian = cv2.Laplacian(eye_region, cv2.CV_64F)
            edge_variance = np.var(laplacian)

            is_dark = mean_intensity < 55
            is_low_contrast = std_intensity < 18
            is_low_texture = edge_variance < 12

            dark_score = max(0, (55 - mean_intensity) / 55) if is_dark else 0
            contrast_score = max(0, (18 - std_intensity) / 18) if is_low_contrast else 0
            texture_score = max(0, (12 - edge_variance) / 12) if is_low_texture else 0

            combined_confidence = (dark_score * 0.4 + contrast_score * 0.35 + texture_score * 0.25)
            is_blocked = (is_dark or is_low_contrast or is_low_texture) and combined_confidence > 0.25

            return is_blocked, float(combined_confidence)

        def check_iris_visibility(face_landmarks, eye_side='left'):
            iris_indices = [468, 469, 470, 471, 472] if eye_side == 'left' else [473, 474, 475, 476, 477]

            iris_points = []
            for idx in iris_indices:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    iris_points.append((lm.x * w, lm.y * h))

            # NOTE: keeping your logic same; this may give false positives in bad light.
            if len(iris_points) < 4:
                return True, 0.5

            distances = []
            center = iris_points[0]
            for pt in iris_points[1:]:
                distances.append(math.sqrt((pt[0] - center[0]) ** 2 + (pt[1] - center[1]) ** 2))

            if not distances:
                return True, 0.5

            avg_dist = np.mean(distances)
            std_dist = np.std(distances)

            is_inconsistent = std_dist > 2.5 or avg_dist < 2.0
            conf = min(1.0, float(std_dist / 5.0)) if is_inconsistent else 0.0
            return is_inconsistent, conf

        left_opacity_blocked, left_opacity_conf = check_eye_region_opacity(left_eye_pos, gray)
        left_iris_blocked, left_iris_conf = check_iris_visibility(face_landmarks, 'left')
        left_blocked = left_opacity_blocked or left_iris_blocked
        left_conf = max(left_opacity_conf, left_iris_conf)

        right_opacity_blocked, right_opacity_conf = check_eye_region_opacity(right_eye_pos, gray)
        right_iris_blocked, right_iris_conf = check_iris_visibility(face_landmarks, 'right')
        right_blocked = right_opacity_blocked or right_iris_blocked
        right_conf = max(right_opacity_conf, right_iris_conf)

        confidence = float((left_conf + right_conf) / 2.0)

    except Exception:
        return False, False, 0.0

    return left_blocked, right_blocked, confidence


# ---------------- Backend Class ----------------
class SpectacleEyeBackend:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.ipd_history = deque(maxlen=HISTORY_LEN)
        self.left_nose_history = deque(maxlen=HISTORY_LEN)
        self.right_nose_history = deque(maxlen=HISTORY_LEN)
        self.nose_line_history = deque(maxlen=HISTORY_LEN)
        self.scale_history = deque(maxlen=HISTORY_LEN)
        self.head_tilt_history = deque(maxlen=HISTORY_LEN)

        self.left_pupil_history = deque(maxlen=PUPIL_HISTORY_LEN)
        self.right_pupil_history = deque(maxlen=PUPIL_HISTORY_LEN)

        self.left_iris_history = deque(maxlen=HISTORY_LEN)
        self.right_iris_history = deque(maxlen=HISTORY_LEN)

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)
        hand_results = self.hands.process(rgb)

        out: Dict[str, Any] = {
            "ok": False,

            # distance
            "distance_cm": None,
            "distance_mm": None,
            "focal_length_px_used": float(FOCAL_LENGTH_PX),

            # pd
            "pd_mm": None,
            "pd_left_mm": None,
            "pd_right_mm": None,

            # pose
            "head_tilt_deg": 0.0,
            "head_tilt_signed_deg": 0.0,

            # iris
            "iris_px": None,
            "scale_mm_per_px": None,

            # eyes open
            "left_eye_open": False,
            "right_eye_open": False,

            # occlusion
            "left_hand_blocking": False,
            "right_hand_blocking": False,
            "left_frame_blocking": False,
            "right_frame_blocking": False,
            "frame_detection_confidence": 0.0,

            # for overlays
            "auto_left_pupil": None,
            "auto_right_pupil": None,

            "warnings": [],
        }

        if not results.multi_face_landmarks:
            out["warnings"].append("No face detected")
            return out

        face_landmarks = results.multi_face_landmarks[0]

        try:
            raw_left = (int(face_landmarks.landmark[468].x * w), int(face_landmarks.landmark[468].y * h))
            raw_right = (int(face_landmarks.landmark[473].x * w), int(face_landmarks.landmark[473].y * h))
            nose_center = (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h))

            # sunglasses
            left_frame_blocking, right_frame_blocking, frame_conf = detect_sunglasses_or_frames(
                frame_bgr, raw_left, raw_right, face_landmarks, w, h
            )
            out["left_frame_blocking"] = bool(left_frame_blocking)
            out["right_frame_blocking"] = bool(right_frame_blocking)
            out["frame_detection_confidence"] = float(frame_conf)

            # ✅ FIXED: hands were swapped wrongly in your file
            left_hand_blocking, right_hand_blocking = check_hands_covering_eyes(
                hand_results, raw_left, raw_right, w, h
            )
            out["left_hand_blocking"] = bool(left_hand_blocking)
            out["right_hand_blocking"] = bool(right_hand_blocking)

            # smooth pupil
            self.left_pupil_history.append(raw_left)
            self.right_pupil_history.append(raw_right)

            auto_left = weighted_average_position(self.left_pupil_history)
            auto_right = weighted_average_position(self.right_pupil_history)

            out["auto_left_pupil"] = auto_left
            out["auto_right_pupil"] = auto_right

            final_left = auto_left
            final_right = auto_right

            # ✅ FIXED: left/right EYE INDICES were swapped in your file
            # MediaPipe standard:
            # LEFT eye: 33, 160, 158, 133, 153, 159
            # RIGHT eye: 362, 385, 387, 263, 373, 386
            left_eye_idx = [33, 160, 158, 133, 153, 159]
            right_eye_idx = [362, 385, 387, 263, 373, 386]

            ear_left = eye_aspect_ratio(face_landmarks.landmark, left_eye_idx, w, h)
            ear_right = eye_aspect_ratio(face_landmarks.landmark, right_eye_idx, w, h)

            left_open = ear_left > 0.20
            right_open = ear_right > 0.20
            out["left_eye_open"] = bool(left_open)
            out["right_eye_open"] = bool(right_open)

            # iris px
            left_iris_px = dist(
                (face_landmarks.landmark[469].x * w, face_landmarks.landmark[469].y * h),
                (face_landmarks.landmark[471].x * w, face_landmarks.landmark[471].y * h),
            )
            right_iris_px = dist(
                (face_landmarks.landmark[474].x * w, face_landmarks.landmark[474].y * h),
                (face_landmarks.landmark[476].x * w, face_landmarks.landmark[476].y * h),
            )

            if left_iris_px > 0 and left_open:
                self.left_iris_history.append(left_iris_px)
            if right_iris_px > 0 and right_open:
                self.right_iris_history.append(right_iris_px)

            iris_candidates = []
            if self.left_iris_history:
                iris_candidates.append(float(np.median(filter_outliers(list(self.left_iris_history)))))
            if self.right_iris_history:
                iris_candidates.append(float(np.median(filter_outliers(list(self.right_iris_history)))))

            iris_px = float(sum(iris_candidates) / len(iris_candidates)) if iris_candidates else 0.0

            # clamp insane iris_px
            if iris_px < 4 or iris_px > 80:
                iris_px = 0.0

            out["iris_px"] = iris_px if iris_px > 0 else None

            # scale
            scale_to_use = None
            if iris_px > 0:
                current_scale = IRIS_REAL_MM / iris_px
                if 0.05 < current_scale < 0.5:
                    self.scale_history.append(current_scale)

            if len(self.scale_history) >= 3:
                scale_to_use = float(np.median(filter_outliers(list(self.scale_history))))
            else:
                scale_to_use = float(sum(self.scale_history) / len(self.scale_history)) if self.scale_history else None

            if (scale_to_use is None) and iris_px > 0:
                scale_to_use = float(IRIS_REAL_MM / iris_px)

            out["scale_mm_per_px"] = scale_to_use

            # Distance
            if iris_px > 0:
                distance_mm = float((IRIS_REAL_MM * float(FOCAL_LENGTH_PX)) / iris_px)

                # clamp insane output
                if 150 <= distance_mm <= 2000:
                    out["distance_mm"] = distance_mm
                    out["distance_cm"] = float(distance_mm / 10.0)

            # Head tilt
            tilt, tilt_signed = calculate_head_tilt_deg(face_landmarks, w, h, self.head_tilt_history)
            out["head_tilt_deg"] = float(tilt)
            out["head_tilt_signed_deg"] = float(tilt_signed)

            if abs(tilt) > HEAD_TILT_LIMIT_DEG:
                out["warnings"].append("Head tilt too high")

            # nose line
            if final_left and final_right:
                eye_line_y = (final_left[1] + final_right[1]) / 2.0
            else:
                eye_line_y = h // 2

            raw_nose_line_point = (nose_center[0], int(eye_line_y))
            self.nose_line_history.append(raw_nose_line_point)
            avg_nose_line_point = (
                int(sum(pt[0] for pt in self.nose_line_history) / len(self.nose_line_history)),
                int(sum(pt[1] for pt in self.nose_line_history) / len(self.nose_line_history)),
            )

            effective_left = left_open and final_left and (not left_hand_blocking) and (not left_frame_blocking)
            effective_right = right_open and final_right and (not right_hand_blocking) and (not right_frame_blocking)

            left_nose_avg = None
            right_nose_avg = None
            ipd_mm_avg = None

            if effective_left and scale_to_use:
                left_to_nose_mm = dist(final_left, avg_nose_line_point) * scale_to_use
                if is_valid_pd(left_to_nose_mm, half_pd=True):
                    self.left_nose_history.append(left_to_nose_mm)
                if self.left_nose_history:
                    left_nose_avg = float(np.median(filter_outliers(list(self.left_nose_history))))

            if effective_right and scale_to_use:
                right_to_nose_mm = dist(final_right, avg_nose_line_point) * scale_to_use
                if is_valid_pd(right_to_nose_mm, half_pd=True):
                    self.right_nose_history.append(right_to_nose_mm)
                if self.right_nose_history:
                    right_nose_avg = float(np.median(filter_outliers(list(self.right_nose_history))))

            if effective_left and effective_right and scale_to_use:
                ipd_mm = dist(final_left, final_right) * scale_to_use
                if is_valid_pd(ipd_mm):
                    self.ipd_history.append(ipd_mm)
                if self.ipd_history:
                    ipd_mm_avg = float(np.median(filter_outliers(list(self.ipd_history))))

            out["pd_mm"] = float(ipd_mm_avg) if ipd_mm_avg else None
            out["pd_left_mm"] = float(left_nose_avg) if left_nose_avg else None
            out["pd_right_mm"] = float(right_nose_avg) if right_nose_avg else None

            if left_frame_blocking or right_frame_blocking:
                out["warnings"].append("Frames/Sunglasses detected")
            if left_hand_blocking or right_hand_blocking:
                out["warnings"].append("Hands detected near eyes")

            out["ok"] = True
            return out

        except Exception as e:
            out["warnings"].append(f"Processing error: {str(e)}")
            return out
