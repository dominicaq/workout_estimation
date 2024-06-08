# workout_estimation
Please use the requirement.txt to install the library dependencies for this
project.

## Running the script for testing
To run the script, format the command like the following:
`python3 main.py "file_path/to/image/image_name.ext"`. 

***Note:*** `.ext` is
the extension of the image.

## Script steps
The process loads an image from the given image path by the user, and extract
the keypoints from it. The keypoints are then fed into a pretrained neural
network to predict the exercise and movement being performed in the image,
determines the correctness and output the result to the user.

## Other applications
API calls could be made so that the model could be laverage for real-time
feedback when exercising.
