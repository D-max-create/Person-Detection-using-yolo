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

# Configuration for Jetson optimization
torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
torch.set_num_threads(1)  # Limit PyTorch threads

# MAVLink and tracking variables
bbox = None  
tracking = False
chan5, chan6, chan7, chan8 = None, None, None, None
chan9 = 0
chan7_prev = None  
chan7_first_received = False
last_bbox_time = time.time()

# Initialize MAVLink connection
master = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600)

def update_mavlink_channels():
    """Optimized MAVLink channel update with reduced sleep time"""
    global chan5, chan6, chan7, chan8, chan9
    while True:
        msg = master.recv_match(type='SERVO_OUTPUT_RAW', blocking=False)
        msg1 = master.recv_match(type='RC_CHANNELS', blocking=False)
        if msg:
            chan5 = msg.servo5_raw
            chan6 = msg.servo6_raw
            chan7 = msg.servo7_raw
        if msg1:
            chan8 = msg1.chan8_raw
            chan9 = msg1.chan9_raw
        time.sleep(0.02)  # Reduced sleep time

# Start MAVLink thread
mavlink_thread = threading.Thread(target=update_mavlink_channels)
mavlink_thread.daemon = True
mavlink_thread.start()

def adjust_brightness_contrast(frame, brightness=0, contrast=0):
    """Optimized brightness/contrast adjustment using LUT"""
    brightness = np.clip(brightness, -100, 100)
    contrast = np.clip(contrast, -100, 100)
    
    # Use LUT for faster brightness/contrast adjustment
    if brightness != 0 or contrast != 0:
        alpha = contrast + 127
        alpha = 131 * alpha / (127 * (131 - alpha))
        gamma = 127 * (1 - alpha)
        
        lut = np.clip(alpha * np.arange(256) + gamma, 0, 255).astype('uint8')
        frame = cv2.LUT(frame, lut)
        
        if brightness > 0:
            frame = cv2.add(frame, np.uint8(brightness))
        elif brightness < 0:
            frame = cv2.subtract(frame, np.uint8(-brightness))
    
    return frame

def get_frames(video_name):
    """Optimized frame generator with hardware acceleration"""
    if not video_name:
        # Use V4L2 backend for webcam
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set higher FPS if supported
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer size to minimize latency
        
        # Warm up camera
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
pid_roll_left = PID(Kp=0.4, Ki=0.08, Kd=0.04, setpoint=0)  
pid_roll_center = PID(Kp=0.1, Ki=0.02, Kd=0.01, setpoint=0)  
pid_roll_right = PID(Kp=0.4, Ki=0.08, Kd=0.04, setpoint=0) 
pid_pitch = PID(Kp=5, Ki=2, Kd=0.5, setpoint=0)

# Set output limits
pid_roll_left.output_limits = (-200, -50)
pid_roll_center.output_limits = (-50,50)
pid_roll_right.output_limits = (50, 200)
pid_pitch.output_limits = (-400, 400)

def send_pwm(master, roll_pwm, pitch_pwm):
    """Optimized PWM sending with RTL check"""
    if chan9 > 1500:
        master.set_mode("RTL")
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
    
    # Load model with optimized settings
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--video_name', default='', type=str, help='videos or image files')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # Load model with half precision if available
    model = ModelBuilder()
    model.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)
    
    # Try using half precision for faster inference
    if cfg.CUDA:
        model = model.half()
    
    tracker = build_tracker(model)

    video_name = args.video_name.split('/')[-1].split('.')[0] if args.video_name else 'webcam'
    frame_generator = get_frames(args.video_name)
    
    # Performance metrics
    frame_count = 0
    start_time = time.time()
    
    for frame in frame_generator:
        frame_count += 1
        
        # Convert to half precision if using CUDA
        if cfg.CUDA:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).to(device).half().permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        frame_height, frame_width = frame.shape[:2]
        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3
        
        # Display RTL status
        rtl_status = "RTL ON" if chan9 > 1500 else "RTL OFF"
        rtl_color = (0, 0, 255) if chan9 > 1500 else (0, 255, 0)
        cv2.putText(frame, f"RTL: {rtl_status}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, rtl_color, 2)
        
        # Adjust brightness/contrast if needed
        if chan8 is not None:
            if chan8 < 1500:
                brightness = int((chan8 - 1000) / 5) - 100
                contrast = 0
            else:
                brightness = 0
                contrast = int((chan8 - 1500) / 5)
            
            frame = adjust_brightness_contrast(frame, brightness, contrast)
        
        # Handle RTL mode
        if chan9 > 1500:
            if tracking:
                master.set_mode("RTL")
                bbox = None
                tracking = False
                pid_roll_left.reset()
                pid_roll_center.reset()
                pid_roll_right.reset()
                pid_pitch.reset()
            continue
        
        # Initialize tracking if chan7 changes
        if chan7_first_received and chan7 != chan7_prev and time.time() - last_bbox_time > 1.5:
            chan7_prev = chan7
            last_bbox_time = time.time()
            
            bbox = (
                max(0, min(chan5 - 20, frame_width - 40)),
                max(0, min(chan6 - 20, frame_height - 40)),
                40,
                40
            )
            
            tracker = build_tracker(model)
            tracker.init(frame, bbox)
            tracking = True
            master.set_mode("FBWB")
        
        # Update tracker
        if tracking and bbox is not None:
            outputs = tracker.track(frame)
            
            if 'bbox' in outputs:
                bbox = list(map(int, outputs['bbox']))
                obj_center_x = bbox[0] + bbox[2] // 2
                obj_center_y = bbox[1] + bbox[3] // 2
                
                # Calculate offsets
                offset_x = (frame_width // 2) - obj_center_x
                offset_y = (frame_height // 2) - obj_center_y
                
                # Select PID controller based on zone
                if obj_center_x < left_boundary:
                    roll_pwm = 1500 + pid_roll_left(offset_x)
                elif obj_center_x > right_boundary:
                    roll_pwm = 1500 + pid_roll_right(offset_x)
                else:
                    roll_pwm = 1500 + pid_roll_center(offset_x)
                
                pitch_pwm = 1500 - pid_pitch(offset_y)
                send_pwm(master, roll_pwm, pitch_pwm)
                
                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), 
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                              (0, 255, 0), 2)
        
        # Display FPS
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow(video_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    print(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")

if __name__ == '__main__':
    main()
