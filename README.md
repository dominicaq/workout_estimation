## Workout Estimator

Please refer to the [`report.pdf`](./report.pdf) for a detailed write-up of our work. To explore the implementations, view our branches.

## Summary
Our project focuses on creating a workout assistant using computer vision to classify and evaluate different exercises and assess users' posture for correctness. We designed a machine learning model that uses key points from an image to identify exercises such as squats, pushups, and deadlifts. The system can predict whether movements involve flexion or extension and assess the correctness of the posture to provide valuable feedback.

The results show the model significantly improves workout quality and safety. Future improvements include expanding the exercise types, refining the correctness assessment, and adding visual overlays for user guidance.

## Libraries
This list includes the core libraries we utilized to develop the project.
- **MediaPipe**: For extracting key points from images.
- **OpenCV**: Used for image and video processing.
- **PyTorch**: For training and running the neural networks.
- **Kaggle Datasets**: Used to obtain datasets for training and testing the model.
- **yt-dlp**: For downloading videos used in the training dataset.
- **ffmpeg**: For trimming and processing video files.
