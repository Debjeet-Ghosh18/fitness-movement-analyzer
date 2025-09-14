import numpy as np
from collections import deque
import json

class ExerciseAnalyzer:
    def __init__(self):
        """Initialize exercise analyzer with tracking variables."""
        self.exercise_count = 0
        self.stage = None
        self.current_exercise = None
        self.angle_history = deque(maxlen=30)  # Store last 30 frames
        self.feedback_messages = []
        
        # Exercise thresholds and patterns
        self.exercise_patterns = {
            'pushup': {
                'key_joints': ['left_elbow', 'right_elbow'],
                'down_threshold': 90,
                'up_threshold': 160,
                'ideal_range': (70, 90),
                'form_checks': ['back_straight', 'full_range']
            },
            'squat': {
                'key_joints': ['left_knee', 'right_knee'],
                'down_threshold': 90,
                'up_threshold': 160,
                'ideal_range': (80, 100),
                'form_checks': ['knees_aligned', 'back_straight']
            },
            'situp': {
                'key_joints': ['left_hip', 'right_hip'],
                'down_threshold': 150,
                'up_threshold': 70,
                'ideal_range': (45, 70),
                'form_checks': ['controlled_movement', 'neck_neutral']
            },
            'jumping_jack': {
                'key_joints': ['left_shoulder', 'right_shoulder'],
                'down_threshold': 30,
                'up_threshold': 150,
                'ideal_range': (150, 180),
                'form_checks': ['synchronized_movement']
            },
            'lunge': {
                'key_joints': ['left_knee', 'right_knee'],
                'down_threshold': 90,
                'up_threshold': 160,
                'ideal_range': (85, 95),
                'form_checks': ['knee_alignment', 'balance']
            },
            'plank': {
                'key_joints': ['left_hip', 'right_hip'],
                'static': True,
                'ideal_range': (170, 180),
                'form_checks': ['back_straight', 'core_engaged']
            }
        }
        
    def detect_exercise(self, angles, landmarks):
        """
        Detect which exercise is being performed based on movement patterns.
        
        Args:
            angles: Dictionary of joint angles
            landmarks: Pose landmarks
        
        Returns:
            Detected exercise name or 'unknown'
        """
        if not angles:
            return 'unknown'
        
        self.angle_history.append(angles)
        
        # Need history to detect patterns
        if len(self.angle_history) < 10:
            return self.current_exercise or 'unknown'
        
        # Check for each exercise pattern
        for exercise, pattern in self.exercise_patterns.items():
            if self._matches_pattern(exercise, pattern):
                self.current_exercise = exercise
                return exercise
        
        return self.current_exercise or 'unknown'
    
    def _matches_pattern(self, exercise_name, pattern):
        """Check if current movement matches exercise pattern."""
        if pattern.get('static', False):
            # For static exercises like plank
            return self._check_static_position(pattern)
        else:
            # For dynamic exercises
            return self._check_dynamic_movement(pattern)
    
    def _check_static_position(self, pattern):
        """Check if maintaining static position."""
        recent_angles = list(self.angle_history)[-10:]
        key_joints = pattern['key_joints']
        
        for angles in recent_angles:
            for joint in key_joints:
                if joint in angles:
                    angle = angles[joint]
                    min_angle, max_angle = pattern['ideal_range']
                    if not (min_angle <= angle <= max_angle):
                        return False
        return True
    
    def _check_dynamic_movement(self, pattern):
        """Check if performing dynamic movement."""
        recent_angles = list(self.angle_history)[-15:]
        key_joints = pattern['key_joints']
        
        # Check for movement range
        for joint in key_joints:
            joint_angles = [a.get(joint, 0) for a in recent_angles if joint in a]
            if len(joint_angles) > 0:
                angle_range = max(joint_angles) - min(joint_angles)
                if angle_range > 40:  # Significant movement detected
                    return True
        return False
    
    def count_repetitions(self, angles, exercise_type):
        """
        Count exercise repetitions based on joint angles.
        
        Args:
            angles: Dictionary of current joint angles
            exercise_type: Type of exercise being performed
        
        Returns:
            Current repetition count
        """
        if exercise_type not in self.exercise_patterns:
            return self.exercise_count
        
        pattern = self.exercise_patterns[exercise_type]
        
        # Skip if static exercise
        if pattern.get('static', False):
            return self.exercise_count
        
        key_joints = pattern['key_joints']
        down_threshold = pattern['down_threshold']
        up_threshold = pattern['up_threshold']
        
        # Check primary joint angle
        for joint in key_joints:
            if joint in angles:
                angle = angles[joint]
                
                # State machine for counting
                if exercise_type in ['pushup', 'squat', 'lunge']:
                    if angle < down_threshold and self.stage != 'down':
                        self.stage = 'down'
                    elif angle > up_threshold and self.stage == 'down':
                        self.stage = 'up'
                        self.exercise_count += 1
                        
                elif exercise_type == 'situp':
                    if angle > down_threshold and self.stage != 'down':
                        self.stage = 'down'
                    elif angle < up_threshold and self.stage == 'down':
                        self.stage = 'up'
                        self.exercise_count += 1
                        
                elif exercise_type == 'jumping_jack':
                    if angle < down_threshold and self.stage != 'down':
                        self.stage = 'down'
                    elif angle > up_threshold and self.stage == 'down':
                        self.stage = 'up'
                        self.exercise_count += 1
                
                break  # Only check first available joint
        
        return self.exercise_count
    
    def evaluate_form(self, angles, exercise_type):
        """
        Evaluate exercise form and provide feedback.
        
        Args:
            angles: Dictionary of joint angles
            exercise_type: Type of exercise
        
        Returns:
            List of feedback messages
        """
        feedback = []
        
        if exercise_type not in self.exercise_patterns:
            return feedback
        
        pattern = self.exercise_patterns[exercise_type]
        ideal_min, ideal_max = pattern['ideal_range']
        
        # Check key joint angles
        for joint in pattern['key_joints']:
            if joint in angles:
                angle = angles[joint]
                
                if self.stage == 'down':
                    if exercise_type in ['pushup', 'squat']:
                        if angle > ideal_max:
                            feedback.append(f"Go deeper! {joint.replace('_', ' ').title()} angle: {angle:.0f}°")
                        elif angle < ideal_min:
                            feedback.append(f"Don't go too low! Risk of injury.")
                    elif exercise_type == 'situp':
                        if angle < ideal_min:
                            feedback.append(f"Come up higher! Target: {ideal_min}°")
        
        # Check form specifics
        if 'back_straight' in pattern['form_checks']:
            if 'left_hip' in angles and 'right_hip' in angles:
                hip_avg = (angles['left_hip'] + angles['right_hip']) / 2
                if exercise_type == 'pushup' and hip_avg < 160:
                    feedback.append("Keep your back straight!")
                elif exercise_type == 'plank' and hip_avg < 170:
                    feedback.append("Don't let your hips sag!")
        
        if 'knees_aligned' in pattern['form_checks']:
            if 'left_knee' in angles and 'right_knee' in angles:
                knee_diff = abs(angles['left_knee'] - angles['right_knee'])
                if knee_diff > 15:
                    feedback.append("Keep knees aligned!")
        
        return feedback
    
    def get_performance_summary(self):
        """Get summary of workout performance."""
        return {
            'exercise': self.current_exercise,
            'repetitions': self.exercise_count,
            'current_stage': self.stage,
            'quality_score': self._calculate_quality_score()
        }
    
    def _calculate_quality_score(self):
        """Calculate form quality score based on consistency."""
        if len(self.angle_history) < 10:
            return 0
        
        # Simple quality metric based on movement consistency
        recent_angles = list(self.angle_history)[-10:]
        consistency_scores = []
        
        for i in range(1, len(recent_angles)):
            if recent_angles[i] and recent_angles[i-1]:
                # Compare angle changes
                for joint in recent_angles[i]:
                    if joint in recent_angles[i-1]:
                        change = abs(recent_angles[i][joint] - recent_angles[i-1][joint])
                        # Smooth changes are good (not too fast, not static)
                        if 1 < change < 20:
                            consistency_scores.append(1)
                        else:
                            consistency_scores.append(0)
        
        if consistency_scores:
            return int(sum(consistency_scores) / len(consistency_scores) * 100)
        return 0
    
    def reset(self):
        """Reset counting and tracking variables."""
        self.exercise_count = 0
        self.stage = None
        self.current_exercise = None
        self.angle_history.clear()
        self.feedback_messages = []