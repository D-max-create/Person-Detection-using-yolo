# export PYTHONPATH=/home/sanjana/pysot/:$PYTHONPATH

# sudo python3 1.py     --config experiments/siamrpn_alex_dwxcorr/config.yaml     --snapshot experiments/siamrpn_alex_dwxcorr/model.pth

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
chan5, chan6, chan7 = None, None, None
chan9 = 0
chan7_prev = None  
chan7_first_received = False  # Track if channel 7 has been received for the first time
last_bbox_time = time.time()  # Prevent rapid re-initialization

# MAVLink connection
master = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600)

'''
def update_mavlink_channels():
    """Continuously updates MAVLink channel values."""
    global chan5, chan6, chan7, chan9
    while True:
        msg = master.recv_match(type='SERVO_OUTPUT_RAW', blocking=True)
        if msg:
            chan5 = msg.servo5_raw
            chan6 = msg.servo6_raw
            chan7 = msg.servo7_raw
            chan9 = msg.servo8_raw
        time.sleep(0.05)  # Prevent excessive CPU usage
'''
'''
def update_mavlink_channels():
    """Continuously updates MAVLink channel values."""
    global chan5, chan6, chan7, chan9
    while True:
        msg = master.recv_match(type='SERVO_OUTPUT_RAW', blocking=True)
        if msg:
            if chan9 > 1500:  # Stop updating MAVLink channels when RTL is active
                continue
            chan5 = msg.servo5_raw
            chan6 = msg.servo6_raw
            chan7 = msg.servo7_raw
            chan9 = msg.servo8_raw
        time.sleep(0.05)  # Prevent excessive CPU usage
 '''       
 
def update_mavlink_channels():
    """Continuously updates MAVLink channel values."""
    global chan5, chan6, chan7, chan9
    while True:
        msg = master.recv_match(type='SERVO_OUTPUT_RAW', blocking=True)
        msg1 = master.recv_match(type='RC_CHANNELS', blocking=True)
        if msg and msg1:
               
            chan5 = msg.servo5_raw
            chan6 = msg.servo6_raw
            chan7 = msg.servo7_raw
            chan9 = msg1.chan9_raw
        time.sleep(0.05)  # Prevent excessive CPU usage
        
# Start MAVLink thread
mavlink_thread = threading.Thread(target=update_mavlink_channels)
mavlink_thread.daemon = True
mavlink_thread.start()

def get_frames(video_name):
    """Generates frames from webcam, video, or image sequence."""
    if not video_name:
        cap = cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
pid_roll = PID(Kp=0.3, Ki=0.06, Kd=0.03, setpoint=0)
pid_pitch = PID(Kp=5, Ki=2, Kd=0.5, setpoint=0)
pid_roll.output_limits = (-200, 200)
pid_pitch.output_limits = (-400, 400)

def send_pwm(master, roll_pwm, pitch_pwm):
    """Sends PWM signals for drone movement, but stops if RTL is engaged."""
    if chan9 > 1500:  # Stop sending PWM when RTL is active
            master.set_mode("RTL")
            print(chan9, 'chan 9 value')
            print(f"RTL detected! Clearing bbox at {time.time()}")
            bbox = None  # CLEAR bbox immediately
            tracking = False
            pid_roll.reset()
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
        #chan9 = 0

        # Stop tracking and remove bbox if RTL is engaged
        if chan9 > 1500:
                
            master.set_mode("RTL")
            print(chan9, '!!!!!chan 9 value')
            print(f"RTL detected! Clearing bbox at {time.time()}")
            bbox = None  # CLEAR bbox immediately
            tracking = False
            pid_roll.reset()
            pid_pitch.reset()
            print("Tracking stopped. Mode set to RTL.")
            #print(roll_pwm, ' RTL  roll')
            #print(pitch_pwm, ' RTL  pitch')
            

        # Track first reception of chan7
        if chan7 is not None and not chan7_first_received:
            chan7_first_received = True  
            chan7_prev = chan7  
            print("First reception of chan7 detected. Ignoring for bbox initialization.")

        # Initialize bbox only if chan7 changes, enough time has passed, and RTL is NOT active
        #	msg2 = master.recv_match(type='RC_CHANNELS', blocking=True)
        #chan9 = msg2.chan9_raw
        if chan7_first_received and chan7 != chan7_prev and time.time() - last_bbox_time > 1.5 and chan9 <= 1500:
            print(f"Channel 7 changed: {chan7_prev} -> {chan7}")
            chan7_prev = chan7  
            last_bbox_time = time.time()  

            # Calculate new bbox
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

            # Reset tracker
            tracker = build_tracker(model)
            tracker.init(frame, bbox)  
            tracking = True  
            master.set_mode("FBWB")
            print(f"New bounding box initialized: {bbox}. Mode set to FBWB.")
            
            '''
            master.mav.command_long_send(
			    master.target_system,
			    master.target_component,
			    mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
			    0,  # Confirmation
			    9,  # Channel 9
			    1100,  # PWM value
			    0, 0, 0, 0, 0  # Unused parameters
			    )

            print(f"New bounding box initialized: {bbox}. Mode set to FBWB. Channel 8 set to 1100.")
            '''
        # Update the tracker if tracking is in progress and RTL is NOT engaged
        if tracking & (bbox != None) & (chan9 <= 1500):
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

                roll_pwm = 1500 + pid_roll(offset_x)
                pitch_pwm = 1500 - pid_pitch(offset_y)

                send_pwm(master, roll_pwm, pitch_pwm)
                print(roll_pwm, '   roll')
                print(pitch_pwm, '   pitch')

        # Show the frame **without bbox** if RTL is active
        if bbox is not None and chan9 <= 1500:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

        cv2.imshow(video_name, frame)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
