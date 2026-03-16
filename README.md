
# Person Detection and Tracking using YOLOv8

This project performs **real-time object detection and tracking** using YOLOv8 and OpenCV.  
The system detects objects such as **people, cars, buses, trucks, and motorbikes** from a webcam feed and allows the user to click on a detected object to start tracking it.

## Features
- Real-time object detection using YOLOv8
- Detects:
  - Person
  - Car
  - Bus
  - Truck
  - Motorbike
- Click on any detected object to start tracking
- Uses CSRT Tracker from OpenCV for accurate tracking
- Reset tracking anytime with a key press

## Technologies Used
- Python
- OpenCV
- YOLOv8 (Ultralytics)

## Installation

### 1. Clone the repository

git clone https://github.com/your-username/Person-Detection-using-yolo.git

cd Person-Detection-using-yolo


### 2. Install required libraries

pip install opencv-python
pip install ultralytics


### 3. Download YOLOv8 model
The project uses the lightweight YOLOv8 nano model:


yolov8n.pt


Place the model file inside the project folder.

## How It Works

1. The webcam captures live video using OpenCV.
2. YOLOv8 detects objects in each frame.
3. Bounding boxes are drawn around detected objects.
4. When the user clicks on a detected object:
   - The system initializes a CSRT tracker.
5. The tracker follows the selected object in real-time.

## Controls

| Key | Action |
|----|----|
| Mouse Click | Select object to track |
| R | Reset tracking |
| ESC | Exit program |

## Run the Project


python webcamera.py


## Example Workflow

1. Run the program.
2. Detected objects appear with green bounding boxes.
3. Click on a detected object.
4. The object will be tracked with a blue bounding box.

## Future Improvements

- Multi-object tracking
- Object counting
- Performance optimization with GPU
- Save tracking results

## Author
