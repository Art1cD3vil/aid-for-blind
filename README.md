Pothole Detection System for the Visually Impaired
Overview
This project aims to assist visually impaired individuals by detecting obstacles such as potholes, stairs, and other potential hazards in their path. The system is built using a custom machine learning model based on YOLOv8 and calculates the distance to detected obstacles using the principle of similar triangles. The setup is embedded in a Raspberry Pi 4, utilizing a Pi Camera to capture real-time footage and providing audio feedback to the user.

Features
Real-time Object Detection: Detects obstacles such as potholes, stairs, and other hazards.
Distance Calculation: Uses the principle of similar triangles to calculate the distance to detected objects.
Audio Feedback: Provides real-time audio alerts to inform the user of detected obstacles and their distance.
Embedded System: Runs on a Raspberry Pi 4 with a Pi Camera for portability and ease of use.
System Architecture
Object Detection:

Custom YOLOv8 model trained to recognize potholes, stairs, and other relevant obstacles.
Model deployed on Raspberry Pi 4 for real-time inference.
Distance Calculation:

Calculates the distance to detected objects using the principle of similar triangles.
Uses the dimensions of detected objects in the image and the known height of the camera.
Audio Output:

Text-to-speech system provides verbal alerts about the type and distance of detected obstacles.
Hardware Requirements
Raspberry Pi 4
Pi Camera
MicroSD Card (32GB recommended)
Power Supply for Raspberry Pi
Speaker or Earphones for Audio Output
Software Requirements
Raspberry Pi OS: Ensure you have the latest version installed.
Python 3.x: Required for running the object detection and distance calculation scripts.
YOLOv8: For object detection.
OpenCV: For image processing and distance calculation.
gTTS (Google Text-to-Speech): For converting text to speech.
PyTorch: For running the YOLOv8 model.
