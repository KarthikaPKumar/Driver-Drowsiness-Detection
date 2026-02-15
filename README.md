# Driver Drowsiness Detection using CNN – Real-Time AI Target Detection

This project demonstrates how computer vision and machine learning can be
combined to process noisy real-time camera data and detect targets.

## Features
- Real-time video processing with OpenCV
- Noise filtering and edge detection
- Machine Learning classification (SVM)
- Live confidence scoring

## How It Works
1. Preprocess frames (grayscale, blur, edges)
2. Extract pixel features
3. Train ML classifier
4. Predict targets in real time

## Run
pip install -r requirements.txt
python train_model.py
python realtime_detection.py
