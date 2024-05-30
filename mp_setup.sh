# Note: I chose the heavy model, we can use other models though. See below for
# other models.
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models

# Function to check if a Python package is installed
is_installed() {
    pip show "$1" > /dev/null 2>&1
}

# Define model URL and target directory
MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
TARGET_DIR="mp-model"
MODEL_NAME="pose_landmarker.task"
MODEL_PATH="$TARGET_DIR/$MODEL_NAME"

# Check if MediaPipe is installed
if is_installed "mediapipe"; then
    echo "MediaPipe is already installed."
else
    echo "Installing MediaPipe..."
    pip install -q mediapipe
    echo "MediaPipe installed."
fi

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR

# Check if the model is already downloaded
if [ -f "$MODEL_PATH" ]; then
    echo "Pose Landmarker model is already downloaded."
else
    echo "Downloading Pose Landmarker model..."
    curl -o $MODEL_PATH -s $MODEL_URL
    echo "Pose Landmarker model downloaded to $MODEL_PATH."
fi
