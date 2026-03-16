from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import cv2
import torch
import numpy as np
from glob import glob
from pymavlink import mavutil
import threading
import time
from simple_pid import PID

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str, help='videos or image files')
args = parser.parse_args()

# Global variables
bbox = None  
tracking = False
chan5, chan6, chan7, chan8 = None, None, None, None
chan9 = 0
chan7_prev = None  
chan7_first_received = False
last_bbox_time = time.time()

# MAVLink connection
master = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600)

def update_mavlink_channels():
    """Continuously updates MAVLink channel values."""
    global chan5, chan6, chan7, chan8, chan9
    while True:
        msg = master.recv_match(type='SERVO_OUTPUT_RAW', blocking=True)
        msg1 = master.recv_match(type='RC_CHANNELS', blocking=True)
        if msg and msg1:
            chan5 = msg.servo5_raw
            chan6 = msg.servo6_raw
            chan7 = msg.servo7_raw
            chan8 = msg1.chan8_raw
            chan9 = msg1.chan9_raw
        time.sleep(0.05)

# Start MAVLink thread
mavlink_thread = threading.Thread(target=update_mavlink_channels)
mavlink_thread.daemon = True
mavlink_thread.start()

def adjust_brightness_contrast(frame, brightness=0, contrast=0):
    """
    Adjust brightness and contrast of the frame.
    brightness: -100 to 100 (0 = no change)
    contrast: -100 to 100 (0 = no change)
    """
    # Normalize brightness and contrast values
    brightness = np.clip(brightness, -100, 100)
    contrast = np.clip(contrast, -100, 100)
    
    # Convert brightness and contrast values to OpenCV parameters
    brightness = int(255 * (brightness / 100))
    contrast = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    
    # Apply brightness and contrast
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        frame = cv2.addWeighted(frame, alpha_b, frame, 0, gamma_b)
    
    if contrast != 0:
        frame = cv2.addWeighted(frame, contrast, frame, 0, 127*(1-contrast))
    
    return frame

def get_frames(video_name):
    """Generates frames from webcam, video, or image sequence."""
    if not video_name:
        cap = cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cv2.namedWindow("webcam", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        for _ in range(5):
            cap.read()  
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith(('avi', 'mp4')):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield cv2.resize(frame, (640, 480))
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, '*.jpg')))
        for img in images:
            frame = cv2.imread(img)
            yield cv2.resize(frame, (640, 480))

# PID Controllers
# Three PID controllers for roll (left, center, right zones)
pid_roll_left = PID(Kp=10, Ki=0.2, Kd=0.2, setpoint=0)  
pid_roll_center = PID(Kp=8, Ki=0.15, Kd=0.1, setpoint=0)  
pid_roll_right = PID(Kp=10, Ki=0.2, Kd=0.2, setpoint=0) 

# Single PID controller for pitch (unchanged)
pid_pitch = PID(Kp=5, Ki=2, Kd=0.5, setpoint=0)

# Set output limits
pid_roll_left.output_limits = (-200, -50)
pid_roll_center.output_limits = (-50,50)
pid_roll_right.output_limits = (50, 200)
pid_pitch.output_limits = (-400, 400)

def send_pwm(master, roll_pwm, pitch_pwm):
    """Sends PWM signals for drone movement, but stops if RTL is engaged."""
    if chan9 > 1500:
            master.set_mode("RTL")
            print(chan9, 'chan 9 value')
            #print(f"RTL detected! Clearing bbox at {time.time()}")
            bbox = None
            tracking = False
            pid_roll_left.reset()
            pid_roll_center.reset()
            pid_roll_right.reset()
            pid_pitch.reset()
            print("Tracking stopped. Mode set to RTL.")
            print("RTL engaged. Stopping PWM commands.")
            return

    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        int(roll_pwm),
        int(pitch_pwm),
        0, 0, 0, 0, 0, 0
    )

def main():
    global bbox, tracking, chan7_prev, chan7_first_received, last_bbox_time
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    model = ModelBuilder()
    model.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    tracker = build_tracker(model)

    video_name = args.video_name.split('/')[-1].split('.')[0] if args.video_name else 'webcam'
    frame_generator = get_frames(args.video_name)

    for frame in frame_generator:
        frame_height, frame_width = frame.shape[:2]
        
        # Add RTL status text to the frame
        rtl_status = "RTL ON" if chan9 > 1500 else "RTL OFF"
        rtl_color = (0, 0, 255) if chan9 > 1500 else (0, 255, 0)
        cv2.putText(frame, f"RTL Status: {rtl_status}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, rtl_color, 2)
        
        # Adjust brightness/contrast based on chan8
        if chan8 is not None:
            # Map chan8 (1000-2000) to brightness (-100 to 100) and contrast (-100 to 100)
            # Split the range: <1500 for brightness, >1500 for contrast
            if chan8 < 1500:
                brightness = int((chan8 - 1000) / 5) - 100  # 1000->-100, 1500->0
                contrast = 0
            else:
                brightness = 0
                contrast = int((chan8 - 1500) / 5)  # 1500->0, 2000->100
            
            frame = adjust_brightness_contrast(frame, brightness, contrast)
            
            # Display brightness/contrast settings
            #cv2.putText(frame, f"Brightness: {brightness}", (20, 80), 
                       #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #cv2.putText(frame, f"Contrast: {contrast}", (20, 120), 
                       #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw vertical zone dividers (for visualization)
        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3

        # Stop tracking and remove bbox if RTL is engaged
        if chan9 > 1500:
            master.set_mode("RTL")
            print(chan9, '!!!!!chan 9 value')
            print(f"RTL detected! Clearing bbox at {time.time()}")
            bbox = None
            tracking = False
            pid_roll_left.reset()
            pid_roll_center.reset()
            pid_roll_right.reset()
            pid_pitch.reset()
            print("Tracking stopped. Mode set to RTL.")

        # Track first reception of chan7
        if chan7 is not None and not chan7_first_received:
            chan7_first_received = True  
            chan7_prev = chan7  
            print("First reception of chan7 detected. Ignoring for bbox initialization.")

        # Initialize bbox only if chan7 changes, enough time has passed, and RTL is NOT active
        if chan7_first_received and chan7 != chan7_prev and time.time() - last_bbox_time > 1.5 and chan9 <= 1500:
            print(f"Channel 7 changed: {chan7_prev} -> {chan7}")
            chan7_prev = chan7  
            last_bbox_time = time.time()  

            bbox = (
                max(0, chan5 - 20),
                max(0, chan6 - 20),
                40,
                40
            )
            bbox = (
                min(bbox[0], frame_width - bbox[2]),
                min(bbox[1], frame_height - bbox[3]),
                bbox[2],
                bbox[3]
            )

            tracker = build_tracker(model)
            tracker.init(frame, bbox)  
            tracking = True  
            master.set_mode("FBWB")
            print(f"New bounding box initialized: {bbox}. Mode set to FBWB.")

        # Update the tracker if tracking is in progress and RTL is NOT engaged
        if tracking and (bbox is not None) and (chan9 <= 1500):
            outputs = tracker.track(frame)

            if 'bbox' in outputs:
                bbox = list(map(int, outputs['bbox']))
                print("Updated BBOX:", bbox)

                obj_center_x = bbox[0] + bbox[2] // 2
                obj_center_y = bbox[1] + bbox[3] // 2

                frame_center_x = frame_width // 2
                frame_center_y = frame_height // 2

                offset_x = frame_center_x - obj_center_x
                offset_y = frame_center_y - obj_center_y

                # Determine which zone the object is in and select appropriate PID controller
                if obj_center_x < left_boundary:  # Left zone
                    roll_pwm = 1500 + pid_roll_left(offset_x)
                    zone = "LEFT"
                elif obj_center_x > right_boundary:  # Right zone
                    roll_pwm = 1500 + pid_roll_right(offset_x)
                    zone = "RIGHT"
                else:  # Center zone
                    roll_pwm = 1500 + pid_roll_center(offset_x)
                    zone = "CENTER"

                pitch_pwm = 1500 - pid_pitch(offset_y)

                # Display current zone on frame
                #cv2.putText(frame, f"Zone: {zone}", (20, 160), 
                           #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                send_pwm(master, roll_pwm, pitch_pwm)
                print(f"Roll PWM: {roll_pwm} (Zone: {zone}), Pitch PWM: {pitch_pwm}")

        # Show the frame with bbox if not in RTL
        if bbox is not None and chan9 <= 1500:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

        cv2.imshow(video_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

