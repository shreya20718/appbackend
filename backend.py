# import cv2
# import mediapipe as mp
# import numpy as np

# # ================= CONFIG =================
# ASSUMED_IPD_MM = 62.0

# # ================= CORE SYSTEM =================

# class SpectacleFrameDetectionSystem:
#     def __init__(self):
#         self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

#     def process_frame(self, frame):
#         h, w = frame.shape[:2]

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = self.face_mesh.process(rgb)

#         if not result.multi_face_landmarks:
#             return {
#                 "detected": False
#             }

#         # 🔥 FRAME DETECTED = TRUE (FAKE MODE)
#         frame_detected = True

#         if not frame_detected:
#             return {"detected": False}

#         # ================= STATIC VALUES =================
#         A_mm = 48.0
#         B_mm = 30.0
#         DBL_mm = 18.0

#         # ================= FAKE BOXES =================
#         lens_w = int(w * 0.18)
#         lens_h = int(h * 0.14)

#         center_y = int(h * 0.45)

#         left_x = int(w * 0.30)
#         right_x = int(w * 0.52)

#         left_box = [left_x, center_y, lens_w, lens_h]
#         right_box = [right_x, center_y, lens_w, lens_h]

#         return {
#             "detected": True,
#             "A_mm": A_mm,
#             "B_mm": B_mm,
#             "DBL_mm": DBL_mm,
#             "left_box": left_box,
#             "right_box": right_box
#         }


# # ================= GLOBAL =================

# spectacle_system = SpectacleFrameDetectionSystem()
import cv2
import mediapipe as mp
import numpy as np

# ================= CONFIG =================
ASSUMED_IPD_MM = 62.0

# ================= CORE SYSTEM =================

class SpectacleFrameDetectionSystem:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        # --------- FACE DETECTION ---------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return {"detected": False}

        # --------- SPECTACLE DETECTION ---------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 60, 130)

        # Eye-line + nose bridge region
        roi = edges[int(h * 0.38):int(h * 0.48),
                    int(w * 0.30):int(w * 0.70)]

        edge_density = np.sum(roi > 0) / roi.size

        # Horizontal frame line detection
        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=40,
            maxLineGap=10
        )

        line_detected = lines is not None

        frame_detected = (edge_density > 0.02) or line_detected

        if not frame_detected:
            return {"detected": False}

        # ================= STATIC VALUES =================
        A_mm = 48.0
        B_mm = 30.0
        DBL_mm = 18.0

        return {
            "detected": True,
            "A_mm": A_mm,
            "B_mm": B_mm,
            "DBL_mm": DBL_mm
        }

# ================= GLOBAL =================

spectacle_system = SpectacleFrameDetectionSystem()