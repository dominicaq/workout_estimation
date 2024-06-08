import argparse
import numpy as np
import pandas as pd
import mediapipe as mp

import torch
import torch.nn.functional as F

from utils import mp_init_detector, mp_process_image, ExercisesModel

exercise_classifier_path = 'model\exercise_model.pth'

exercise_mapping = {0: 'pushup', 1: 'squat', 2: 'deadlift'}
movement_mapping = {0: 'extension', 1: 'flexion', 2: 'other'}


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

    has_cuda = torch.cuda.is_available()

    return model.to('cuda' if has_cuda else 'cpu')


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

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enter path to your image')    
    parser.add_argument('image_path', type=str, help='Enter the path to your image...')
    
    args = parser.parse_args()

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

