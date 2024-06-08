import argparse
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F

from utils import mp_init_detector, mp_process_image, ExercisesModel

exercise_classifier_path = 'model\exercise_model.pth'

exercise_mapping = {0: 'pushup', 1: 'squat', 2: 'deadlift'}
movement_mapping = {0: 'extension', 1: 'flexion', 2: 'other'}
correctness_mapping = {0: 'incorrect', 1: 'correct'}

def mp_init_detector(model_path='pose_landmarker.task'):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    return vision.PoseLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image




def preprocess_keypoints(keypoints):
    # Extract x, y, z from each keypoint
    keypoints_array = np.array([[kp.x, kp.y, kp.z] for kp in keypoints.values()]).astype(np.float32)
    keypoints_flat = keypoints_array.flatten()
    keypoints_tensor = torch.tensor(keypoints_flat).unsqueeze(0)  # Add batch dimension
    return keypoints_tensor


# Loads the classifier model for prediction
def load_model(model_path):
    # input_size 33 keypoints* 3 (x, y, z), 
    # number of exercises and movements 3, 
    # correctness output size 2
    model = ExercisesModel(33 * 3, 3, 3, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model.to('cpu')


# Extract keypoints from an image
def extract_keypoints(src_img):
    mp_detector = mp_init_detector()
    mp_image = mp.Image.create_from_file(src_img)
    
    _, key_points = mp_process_image(
        mp_detector,
        mp_image
    )

    return preprocess_keypoints(key_points)

# Evaluates the model, given the keypoints, model predict correctness,
# movement and exercise.
def evaluate_image(model, keypoints):
    model.eval()

    with torch.no_grad():
        exercise, movement, correctness = model(keypoints)

        exercise = torch.argmax(F.softmax(exercise, dim=1), dim=1)
        movement = torch.argmax(F.softmax(movement, dim=1), dim=1)

        correctness = (torch.sigmoid(correctness) > 0.5).int()
        correctness = torch.argmax(correctness, dim=1)

    return exercise.item(), movement.item(), correctness.item()

def display_image(image_path, predicted_exercise, predicted_movement, predicted_correctness, detector):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image.create_from_file(image_path)
    detection_result, body_parts = mp_process_image(detector, mp_image)
    annotated_image = draw_landmarks_on_image(img_rgb, detection_result)
    plt.figure(figsize=(5, 5))
    plt.imshow(annotated_image)
    plt.title(f'Pred: Exercise: {exercise_mapping[predicted_exercise]}, '
              f'Movement: {movement_mapping[predicted_movement]}, '
              f'Correctness: {correctness_mapping[predicted_correctness]}')
    plt.axis('off')
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enter path to your image')    
    parser.add_argument('image_path', type=str, help='Enter the path to your image...')
    
    args = parser.parse_args()

    # Initialize the MediaPipe detector
    detector = mp_init_detector()

    # Extracting the keypoints from the image. Used by classifier
    # for prediction
    keypoints = extract_keypoints(args.image_path)

    # # Loading the model for prediction
    classifer = load_model(exercise_classifier_path)

    # Model inference based on image input
    exercise, movement, correctness = evaluate_image(classifer, keypoints)

    print("\n\nMODEL PREDICTION: \n")
    print("Exercise type: ", exercise_mapping[exercise])
    print("Movement type: ", movement_mapping[movement])
    print("Correctness: ",   correctness)

     # Display the annotated image with the predicted labels
    display_image(
        args.image_path, 
        exercise, 
        movement, 
        correctness, 
        detector
    )
