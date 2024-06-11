# workout_estimation

Please use the `requirements.txt` to install the library dependencies for this project.

## Running the Script for Testing

To run the script, format the command as follows:
`python3 main.py "file_path/to/image/image_name.ext"`.

***Note:*** `.ext` is the extension of the image. There are sample images in `sample-frames`.

## Script Steps

The process loads an image from the user-provided image path, extracts the keypoints from it, and feeds the keypoints into a pretrained neural network. The network predicts the exercise and movement being performed in the image, determines the correctness, and outputs the result to the user.

## Other Applications

API calls can be made so that the model can be leveraged for real-time feedback when exercising.

## Data

The `data_processing` directory contains various scripts to gather and preprocess the data.

`filter_data.py` reads the Kinetics-700-2020 dataset annotation files and extracts the desired exercise classes (e.g., squats, deadlifting, push-ups). These annotations include the names of the exercises, the IDs for the YouTube videos where the exercises are performed, and the starting and ending times.

To learn more about the Kinetics-700-2020 dataset, please visit [Kinetics Dataset](https://github.com/cvdfoundation/kinetics-dataset).

The `download_frames.py` script downloads the YouTube videos, trims the videos to the specified start and end times, and extracts the frames to create a dataset.

The `clean_dataset.py` script applies some visibility criteria to clean up the data.

Finally, the `annotate_images.ipynb` notebook finds relationships between keypoints and their corresponding angles, applying calculations to annotate our data in terms of movement and correctness. The final CSV files, containing `image_path`, `exercise`, `keypoints`, `movement`, and `correctness` labels, can be found in the `datasets` folder.

## Models Exploration

Our first model approach, detailed in `Exercise_Correctness_Using_CNN.ipynb`, was trained using a CNN to determine exercise correctness. However, this approach is computationally expensive, and due to the variety of images, the model had difficulty predicting correctness.

Our second model approach, `Exercise_Classification_Using_NN.ipynb`, is purely an exercise classification model for squats, deadlifts, and push-ups. We decided to use keypoints because they capture only the essential information needed for our task while removing irrelevant information. Furthermore, processing keypoints is less computationally expensive. The model classifies exercises with great accuracy; however, it doesn't provide correctness information.

Our final approach and the one of choice for this project, `Multi-variable_Exercise_Classification.ipynb`, creates a multi-variable model that not only classifies the exercise type, but it also provides information about type of movement (extension, flexion, other) and correctness (correct, incorrect). We achieved **88.8% exercise classification accuracy**, **78.3% movement classification accuracy**, and **83.5% correctness classification accuracy**. The model can be found in `model\exercise_model.pth`. o test this model, go to [Running the script for testing](#running-the-script-for-testing).

## Contributions

Jose Gavidia: Data Collection and preprocessing, Multi-variable exercise classification, annotate_images.ipynb
jgavidiapaz@ucdavis.edu

Billy Ouattara: Data Collection and preprocessing, Exercise correctness using CNN, main.py
btouattara@ucdavis.edu

Dominic Quintero: Data Collection and preprocessing, Exercise classification using NN, posedata_set.py
daqquintero@ucdavis.edu
