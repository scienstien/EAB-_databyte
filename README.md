\# Real-Time Emotion Detection with Dynamic Backgrounds



A real-time computer vision system that detects faces, infers human emotions, replaces the background dynamically based on stable emotional states, and plays contextual audio feedback. Built using OpenCV, DeepFace, and the MediaPipe Tasks API.



This project demonstrates a full real-time ML pipeline: video capture → face detection → emotion analysis → temporal smoothing → semantic segmentation → background compositing → UI + audio feedback.



---



\## Features



\- Real-time webcam capture

\- Face detection using MediaPipe Tasks API (BlazeFace)

\- Emotion recognition using DeepFace (optional custom FER model support)

\- Temporal smoothing of emotion probabilities

\- Dynamic background replacement using semantic segmentation

\- Emotion-based audio feedback (plays sound on stable transitions)

\- Heads-up display (emotion, probabilities, FPS)

\- Modular, extensible project structure



---



\## Tech Stack



\- Python 3.10+

\- OpenCV

\- MediaPipe Tasks API

\- DeepFace

\- TensorFlow / Keras

\- NumPy

\- pygame (for audio feedback)



---



\## Project Structure



Real-Time-Emotion-Detection-with-OpenCV-DeepFace/

│

├── main.py

├── requirements.txt

├── README.md

├── CHANGELOG.md

│

├── src/

│ ├── core/

│ │ ├── analyzer.py

│ │ ├── background\_generator.py

│ │ ├── camera.py

│ │ └── face\_detector.py

│ ├── ui/

│ │ └── visualizer.py

│ └── utils/

│ └── fps\_counter.py

│

├── models/

│ ├── blaze\_face\_short\_range.tflite

│ └── selfie\_segmenter.tflite

│

└── backgrounds/

├── happy.jpeg

├── sad.jpeg

├── angry.jpeg

├── neutral.jpeg

├── fear.jpeg

└── surprise.jpeg







---



\## Setup



\### 1. Clone the repository



```bash

git clone https://github.com/your-username/your-repo-name.git

cd Real-Time-Emotion-Detection-with-OpenCV-DeepFace



python 3.10 -m venv venv

./venv/Scripts/activate

./venv/Scripts/python.exe -m pip install -r requirements.txt

./venv/Scripts/python.exe main.py

```

How It Works



1. Capture frames from webcam



2\. Detect face using MediaPipe Tasks FaceDetector



3\. Analyze emotions using DeepFace



4\. Smooth emotion probabilities over time



5\. Segment foreground (person) using MediaPipe ImageSegmenter



6\. Replace background based on stable emotion



7\. Play sound on emotion/background transition



8\. Render UI overlay and FPS



Known Issues



MediaPipe Tasks API may throw harmless cleanup warnings on exit



Performance depends on CPU; segmentation is computationally heavy



Single-person focus (multi-person support not yet implemented)



Future Improvements



1. Multi-face emotion tracking



2\. GPU acceleration



3\. Emotion timeline logging



4\. Configurable background themes



5\. Emotion-based music instead of single sound cue



6\. Model fine-tuning for emotion classification

