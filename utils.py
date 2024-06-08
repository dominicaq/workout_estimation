import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ExercisesModel(nn.Module):
    def __init__(self, input_size, num_exercises, num_movements, num_correctness):
        super(ExercisesModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.exercise_head = nn.Linear(64, num_exercises)
        self.movement_head = nn.Linear(64, num_movements)
        self.correctness_head = nn.Linear(64, num_correctness)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        exercise_output = self.exercise_head(x)
        movement_output = self.movement_head(x)
        correctness_output = self.correctness_head(x)
        return exercise_output, movement_output, correctness_output

def mp_init_detector(model_path='pose_landmarker.task'):
    # STEP 2: Create a PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    return vision.PoseLandmarker.create_from_options(options)

def mp_process_image(detector, mp_img):
    # Detect features in image
    detection_result = detector.detect(mp_img)

    # Associate enums with string values
    body_parts_dict = {
        'nose': mp.solutions.pose.PoseLandmark.NOSE,
        'left_eye_inner': mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER,
        'left_eye': mp.solutions.pose.PoseLandmark.LEFT_EYE,
        'left_eye_outer': mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER,
        'right_eye_inner': mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER,
        'right_eye': mp.solutions.pose.PoseLandmark.RIGHT_EYE,
        'right_eye_outer': mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER,
        'left_ear': mp.solutions.pose.PoseLandmark.LEFT_EAR,
        'right_ear': mp.solutions.pose.PoseLandmark.RIGHT_EAR,
        'mouth_left': mp.solutions.pose.PoseLandmark.MOUTH_LEFT,
        'mouth_right': mp.solutions.pose.PoseLandmark.MOUTH_RIGHT,
        'left_shoulder': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        'right_shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        'left_elbow': mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        'right_elbow': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
        'left_wrist': mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        'right_wrist': mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
        'left_pinky': mp.solutions.pose.PoseLandmark.LEFT_PINKY,
        'right_pinky': mp.solutions.pose.PoseLandmark.RIGHT_PINKY,
        'left_index': mp.solutions.pose.PoseLandmark.LEFT_INDEX,
        'right_index': mp.solutions.pose.PoseLandmark.RIGHT_INDEX,
        'left_thumb': mp.solutions.pose.PoseLandmark.LEFT_THUMB,
        'right_thumb': mp.solutions.pose.PoseLandmark.RIGHT_THUMB,
        'left_hip': mp.solutions.pose.PoseLandmark.LEFT_HIP,
        'right_hip': mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        'left_knee': mp.solutions.pose.PoseLandmark.LEFT_KNEE,
        'right_knee': mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        'left_ankle': mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        'right_ankle': mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
        'left_heel': mp.solutions.pose.PoseLandmark.LEFT_HEEL,
        'right_heel': mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
        'left_foot_index': mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
        'right_foot_index': mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX
    }

    # Create a dictonary so we can index into body parts
    # EX Usage: body_parts['ankle_left]
    body_parts = {}
    if detection_result.pose_landmarks:
        for pose_landmarks in detection_result.pose_landmarks:
            for part_name, part_enum in body_parts_dict.items():
                body_parts[part_name] = pose_landmarks[part_enum]

    return detection_result, body_parts