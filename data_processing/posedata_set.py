import torch
from torch.utils.data import Dataset
import pandas as pd

class CustomPoseDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.exercise_types = {exercise: idx for idx, exercise in enumerate(self.data['exercise_type'].unique())}
        # First column = type, 1: = data
        self.keypoint_columns = self.data.columns[1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        keypoints = []
        for col in self.keypoint_columns:
            # Split the string into individual float values and extend the keypoints list
            keypoints.extend([float(x) for x in row[col].split(',')])

        keypoints = torch.tensor(keypoints, dtype=torch.float32)  # Convert to tensor

        exercise_type = row['exercise_type']
        label = self.exercise_types[exercise_type]

        return keypoints, label
