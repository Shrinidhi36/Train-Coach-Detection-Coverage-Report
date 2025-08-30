# Train-Coach-Detection-Coverage-Report
**Project Overview**
This project processes train videos to automatically detect the number of coaches and generate a coverage report with representative frames from each coach. It combines video processing, frame extraction, and object detection using YOLO (You Only Look Once) to identify doors and their states.

Features 
**Coach Detection**
Splits a side-view train video into individual coach segments.
Counts the total number of coaches in the train. 
**Frame Extraction**
Extracts representative frames from each coach video (front, middle, rear).
Saves frames with descriptive names in their respective coach folders. 
**Component Detection**
Detects doors and classifies them as: 
Door 
Door Open 
Door Closed 
Annotates images with bounding boxes and labels.
**Report Generation**
Generates PDF and HTML reports with selected frames for each coach.
Provides a visual summary covering the entire train length.

**Technologies Used** 
Python 3.10+ 
OpenCV – for video processing and frame extraction 
NumPy & SciPy – for image and signal processing (coach segmentation) 
YOLO (Ultralytics) – for object detection 
ReportLab – for generating PDF reports 
HTML/CSS – for generating HTML reports

**Setup instructions**
1. Download the repositery.
2. Install the required packages- pip install opencv-python-headless numpy scipy reportlab tqdm ultralytics
3. Place your train videos in the project root.
4. Run the main script- python main.py
5.Output reports and extracted images will be saved in the output/ folder.
