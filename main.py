import torch
import argparse
import numpy as np
import pandas as pd
import mediapipe as mp

from utils import mp_init_detector, mp_process_image, PoseClassifier

exercise_classifier_path = 'model\pose_classifier.pth'

def preprocess_keypoints(keypoints):
    # Extract x, y, z, and visibility from each keypoint
    keypoints_array = np.array([[kp.x, kp.y, kp.z, kp.visibility] for kp in keypoints.values()]).astype(np.float32)
    keypoints_flat = keypoints_array.flatten()
    keypoints_tensor = torch.tensor(keypoints_flat).unsqueeze(0)  # Add batch dimension
    return keypoints_tensor


# Loads the classifier model for prediction
def load_model(model_path):
    model = PoseClassifier(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


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
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enter path to your image')    
    parser.add_argument('image_path', type=str, help='Enter the path to your image...')
    
    args = parser.parse_args()

    # Extracting the keypoints from the image. Used by classifier
    # for prediction
    keypoints = extract_keypoints(args.image_path)

    print(keypoints)

    # # Loading the model for prediction
    # classifer = load_model(exercise_classifier_path)

