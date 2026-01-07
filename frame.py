import cv2
import mediapipe as mp
import numpy as np

class SpectaclesDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key landmarks for spectacles detection
        # Eye region landmarks
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        # Temple/ear region (where frame arms rest)
        self.left_temple = [234, 127, 162]
        self.right_temple = [454, 356, 389]
        
        # Bridge of nose landmarks
        self.nose_bridge = [6, 168, 197, 195]
        
    def detect_spectacles(self, frame):
        """
        Main detection function that returns True only for clear prescription glasses.
        Returns False for sunglasses, fancy glasses, or no glasses.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False
        
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        
        # Convert landmarks to pixel coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        # Run all detection checks
        has_frame_edges = self._detect_frame_edges(frame, landmarks)
        has_bridge = self._detect_bridge(frame, landmarks)
        not_dark = self._check_not_sunglasses(frame, landmarks)
        no_fancy_patterns = self._check_no_fancy_patterns(frame, landmarks)
        
        # All conditions must be met for clear spectacles
        is_clear_spectacles = (
            has_frame_edges and 
            has_bridge and 
            not_dark and 
            no_fancy_patterns
        )
        
        return is_clear_spectacles
    
    def _detect_frame_edges(self, frame, landmarks):
        """Detect if there are visible frame edges around eyes - enhanced for distance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        frame_detected = False
        edge_scores = []
        
        for eye_indices in [self.left_eye_indices, self.right_eye_indices]:
            # Get eye region
            eye_points = [landmarks[i] for i in eye_indices]
            x_coords = [p[0] for p in eye_points]
            y_coords = [p[1] for p in eye_points]
            
            # Larger margin to capture frames from distance
            margin = 25
            x_min = max(0, min(x_coords) - margin)
            x_max = min(gray.shape[1], max(x_coords) + margin)
            y_min = max(0, min(y_coords) - margin)
            y_max = min(gray.shape[0], max(y_coords) + margin)
            
            eye_region = gray[y_min:y_max, x_min:x_max]
            
            if eye_region.size == 0:
                continue
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(eye_region, (3, 3), 0)
            
            # Multi-scale edge detection with lower thresholds for distance
            edges1 = cv2.Canny(blurred, 20, 60)  # More sensitive
            edges2 = cv2.Canny(blurred, 30, 90)  # Medium sensitivity
            
            # Combine edge detections
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Apply morphological operations to connect nearby edges
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size
            edge_scores.append(edge_density)
            
            # Lower threshold for distance detection
            if edge_density > 0.015:  # Reduced from 0.03
                frame_detected = True
        
        # Alternative check: if both eyes show some edges, likely spectacles
        if len(edge_scores) == 2 and all(score > 0.01 for score in edge_scores):
            frame_detected = True
        
        return frame_detected
    
    def _detect_bridge(self, frame, landmarks):
        """Detect bridge of spectacles over nose - enhanced for distance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Get nose bridge region
        bridge_points = [landmarks[i] for i in self.nose_bridge]
        x_coords = [p[0] for p in bridge_points]
        y_coords = [p[1] for p in bridge_points]
        
        # Larger margin for distance
        margin = 15
        x_min = max(0, min(x_coords) - margin)
        x_max = min(gray.shape[1], max(x_coords) + margin)
        y_min = max(0, min(y_coords) - margin)
        y_max = min(gray.shape[0], max(y_coords) + margin)
        
        bridge_region = gray[y_min:y_max, x_min:x_max]
        
        if bridge_region.size == 0:
            return False
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(bridge_region, (3, 3), 0)
        
        # More sensitive edge detection
        edges = cv2.Canny(blurred, 25, 80)
        
        # Dilate to connect edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        edge_density = np.sum(edges > 0) / edges.size
        
        # Lower threshold for distance
        return edge_density > 0.02  # Reduced from 0.04
    
    def _check_not_sunglasses(self, frame, landmarks):
        """Check that the lenses are NOT dark (ruling out sunglasses) - enhanced"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive histogram equalization for better brightness detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        eye_regions_brightness = []
        
        for eye_indices in [self.left_eye_indices, self.right_eye_indices]:
            eye_points = [landmarks[i] for i in eye_indices]
            x_coords = [p[0] for p in eye_points]
            y_coords = [p[1] for p in eye_points]
            
            x_min = max(0, min(x_coords))
            x_max = min(gray.shape[1], max(x_coords))
            y_min = max(0, min(y_coords))
            y_max = min(gray.shape[0], max(y_coords))
            
            eye_region = gray[y_min:y_max, x_min:x_max]
            
            if eye_region.size > 0:
                avg_brightness = np.mean(eye_region)
                eye_regions_brightness.append(avg_brightness)
        
        if not eye_regions_brightness:
            return False
        
        # If average brightness is too low, likely sunglasses
        avg_brightness = np.mean(eye_regions_brightness)
        
        # Adjusted threshold with CLAHE normalization
        return avg_brightness > 60  # Adjusted from 70
    
    def _check_no_fancy_patterns(self, frame, landmarks):
        """Check for absence of fancy patterns, excessive colors, or decorations"""
        # Get overall eye region
        all_eye_indices = self.left_eye_indices + self.right_eye_indices
        eye_points = [landmarks[i] for i in all_eye_indices]
        x_coords = [p[0] for p in eye_points]
        y_coords = [p[1] for p in eye_points]
        
        margin = 20
        x_min = max(0, min(x_coords) - margin)
        x_max = min(frame.shape[1], max(x_coords) + margin)
        y_min = max(0, min(y_coords) - margin)
        y_max = min(frame.shape[0], max(y_coords) + margin)
        
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0:
            return True
        
        # Convert to HSV to check color saturation
        hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # High saturation indicates colorful/fancy glasses
        avg_saturation = np.mean(saturation)
        
        # Clear glasses should have low saturation (mostly transparent/neutral)
        return avg_saturation < 60
    
    def visualize_detection(self, frame, is_spectacles):
        """Draw detection result on frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Draw key landmarks
            for idx in (self.left_eye_indices + self.right_eye_indices + 
                       self.nose_bridge):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Display result
        color = (0, 255, 0) if is_spectacles else (0, 0, 255)
        text = "Clear Spectacles: YES" if is_spectacles else "Clear Spectacles: NO"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color, 2)
        
        return frame


# Main execution
def main():
    detector = SpectaclesDetector()
    cap = cv2.VideoCapture(0)
    
    print("Starting spectacles detection...")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect spectacles
        is_wearing_spectacles = detector.detect_spectacles(frame)
        
        # Visualize
        output_frame = detector.visualize_detection(frame, is_wearing_spectacles)
        
        cv2.imshow('Clear Spectacles Detection', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()