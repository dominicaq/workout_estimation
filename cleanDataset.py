import os
import shutil
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return np.array([])

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if not result.pose_landmarks:
        return np.array([])

    keypoints = []
    for landmark in result.pose_landmarks.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return np.array(keypoints)




def is_good_benching_image(keypoints, vis_threshold=0.5):
    # Define required keypoints for visibility check
    upperbody_keypoints = {
        'shoulders': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        'elbows': [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW],
        'wrists': [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]
    }
    
     # Check visibility of at least one keypoint per upper body part
    for part, keypoints_list in upperbody_keypoints.items():
        if not any(keypoints[kp.value][3] >= vis_threshold for kp in keypoints_list):
            return False
    
    return True




def is_good_squat_or_deadlift_image(keypoints, vis_threshold=0.5):
    # Define required keypoints for visibility check
    lowerbody_keypoints = {
        'hips': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
        'knees': [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE],
        'ankles': [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]
    }
    
    # Check visibility of at least one keypoint per lower body part
    for part, keypoints_list in lowerbody_keypoints.items():
        if not any(keypoints[kp.value][3] >= vis_threshold for kp in keypoints_list):
            return False

    # Upper body keypoints to ensure general visibility of posture
    upperbody_keypoints = {
        'shoulders': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
    }
    
    for part, keypoints_list in upperbody_keypoints.items():
        if not any(keypoints[kp.value][3] >= vis_threshold for kp in keypoints_list):
            return False
    
    return True



def is_good_pushup_image(keypoints, vis_threshold=0.5):
    # Define required keypoints for visibility check
    upperbody_keypoints = {
        'shoulders': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        'elbows': [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW],
        'wrists': [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]
    }
    
    # Check visibility of at least one keypoint per upper body part
    for part, keypoints_list in upperbody_keypoints.items():
        if not any(keypoints[kp.value][3] >= vis_threshold for kp in keypoints_list):
            return False

    # Lower body keypoints to ensure general visibility of posture
    lowerbody_keypoints = {
        'hips': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
        'ankles': [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]
    }

    for part, keypoints_list in lowerbody_keypoints.items():
        if not any(keypoints[kp.value][3] >= vis_threshold for kp in keypoints_list):
            return False
    
    return True


def clean_data(source_dir, target_dir, vis_threshold=0.5, exercise_check_fn=None):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for img_name in os.listdir(source_dir):
        img_path = os.path.join(source_dir, img_name)
        keypoints = extract_keypoints(img_path)
        if keypoints.size > 0 and exercise_check_fn(keypoints, vis_threshold):
            shutil.copy(img_path, target_dir)
            print(f"Copied {img_path} to {target_dir}")
        else:
            print(f"Filtered out {img_path}")


# Clean the data
clean_data(r"frames\train\bench pressing", r"cleaned_frames\train\bench pressing", exercise_check_fn=is_good_benching_image)
clean_data(r"frames\val\bench pressing", r"cleaned_frames\val\bench pressing", exercise_check_fn=is_good_benching_image)

clean_data(r"frames\train\deadlifting", r"cleaned_frames\train\deadlifting", exercise_check_fn=is_good_squat_or_deadlift_image)
clean_data(r"frames\val\deadlifting", r"cleaned_frames\val\deadlifting", exercise_check_fn=is_good_squat_or_deadlift_image)

clean_data(r"frames\train\squat", r"cleaned_frames\train\squat", exercise_check_fn=is_good_squat_or_deadlift_image)
clean_data(r"frames\val\squat", r"cleaned_frames\val\squat", exercise_check_fn=is_good_squat_or_deadlift_image)

clean_data(r"frames\train\push up", r"cleaned_frames\train\push up", exercise_check_fn=is_good_pushup_image)
clean_data(r"frames\val\push up", r"cleaned_frames\val\push up", exercise_check_fn=is_good_pushup_image)

