import cv2
import mediapipe as mp
import numpy as np
from utils.angle_calculator import calculate_angle, calculate_distance

class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe pose detection.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_pose(self, image, draw=True):
        """
        Detect pose landmarks in image.
        
        Args:
            image: Input image (BGR)
            draw: Whether to draw landmarks on image
        
        Returns:
            image: Image with landmarks drawn (if draw=True)
            landmarks: List of pose landmarks
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Detect pose
        results = self.pose.process(image_rgb)
        
        # Convert back to BGR
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        landmarks = []
        if results.pose_landmarks:
            # Draw landmarks
            if draw:
                self.mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
            
            # Extract landmarks
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return image, landmarks
    
    def get_joint_angles(self, landmarks, image_shape):
        """
        Calculate important joint angles for exercise analysis.
        
        Args:
            landmarks: List of pose landmarks
            image_shape: Shape of the image (height, width)
        
        Returns:
            Dictionary of joint angles
        """
        if not landmarks:
            return {}
        
        h, w = image_shape[:2]
        angles = {}
        
        # Convert normalized coordinates to pixel coordinates
        coords = [[int(lm[0] * w), int(lm[1] * h)] for lm in landmarks]
        
        # Define joint indices (MediaPipe pose landmarks)
        joints = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        # Calculate elbow angles
        if all(landmarks[joints[j]][3] > 0.5 for j in ['left_shoulder', 'left_elbow', 'left_wrist']):
            angles['left_elbow'] = calculate_angle(
                coords[joints['left_shoulder']],
                coords[joints['left_elbow']],
                coords[joints['left_wrist']]
            )
        
        if all(landmarks[joints[j]][3] > 0.5 for j in ['right_shoulder', 'right_elbow', 'right_wrist']):
            angles['right_elbow'] = calculate_angle(
                coords[joints['right_shoulder']],
                coords[joints['right_elbow']],
                coords[joints['right_wrist']]
            )
        
        # Calculate knee angles
        if all(landmarks[joints[j]][3] > 0.5 for j in ['left_hip', 'left_knee', 'left_ankle']):
            angles['left_knee'] = calculate_angle(
                coords[joints['left_hip']],
                coords[joints['left_knee']],
                coords[joints['left_ankle']]
            )
        
        if all(landmarks[joints[j]][3] > 0.5 for j in ['right_hip', 'right_knee', 'right_ankle']):
            angles['right_knee'] = calculate_angle(
                coords[joints['right_hip']],
                coords[joints['right_knee']],
                coords[joints['right_ankle']]
            )
        
        # Calculate hip angles
        if all(landmarks[joints[j]][3] > 0.5 for j in ['left_shoulder', 'left_hip', 'left_knee']):
            angles['left_hip'] = calculate_angle(
                coords[joints['left_shoulder']],
                coords[joints['left_hip']],
                coords[joints['left_knee']]
            )
        
        if all(landmarks[joints[j]][3] > 0.5 for j in ['right_shoulder', 'right_hip', 'right_knee']):
            angles['right_hip'] = calculate_angle(
                coords[joints['right_shoulder']],
                coords[joints['right_hip']],
                coords[joints['right_knee']]
            )
        
        return angles
    
    def release(self):
        """Release resources."""
        self.pose.close()