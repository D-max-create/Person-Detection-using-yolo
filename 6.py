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

# Enable cuDNN autotuner for better GPU performance
torch.backends.cudnn.benchmark = True  
torch.set_num_threads(1)  # Limit PyTorch CPU threads

class MAVLinkHandler:
    def __init__(self, port='/dev/ttyTHS1', baud=57600):
        self.master = mavutil.mavlink_connection(port, baud=baud)
        self.chan5, self.chan6, self.chan7, self.chan8, self.chan9 = None, None, None, None, 0
        self._running = True
        self.thread = threading.Thread(target=self._update_channels)
        self.thread.daemon = True
        self.thread.start()

    def _update_channels(self):
        """Threaded MAVLink channel updates with minimal delay."""
        while self._running:
            msg = self.master.recv_match(type='SERVO_OUTPUT_RAW', blocking=False)
            msg1 = self.master.recv_match(type='RC_CHANNELS', blocking=False)
            if msg:
                self.chan5 = msg.servo5_raw
                self.chan6 = msg.servo6_raw
                self.chan7 = msg.servo7_raw
            if msg1:
                self.chan8 = msg1.chan8_raw
                self.chan9 = msg1.chan9_raw
            time.sleep(0.02)  # Reduced delay for faster updates

    def send_pwm(self, roll_pwm, pitch_pwm):
        """Send PWM commands only if not in RTL mode."""
        if self.chan9 <= 1500:  # Only send if RTL is not active
            self.master.mav.rc_channels_override_send(
                self.master.target_system,
                self.master.target_component,
                int(roll_pwm),
                int(pitch_pwm),
                0, 0, 0, 0, 0, 0
            )

    def stop(self):
        """Cleanup thread on exit."""
        self._running = False
        self.thread.join()
        
        
class VideoProcessor:
    def __init__(self, source=None):
        self.source = source
        self.cap = None
        self._init_capture()

    def _init_capture(self):
        """Initialize video capture with V4L2 backend for low latency."""
        if not self.source:  # Webcam
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Small buffer = lower latency
        elif self.source.endswith(('avi', 'mp4')):  # Video file
            self.cap = cv2.VideoCapture(self.source)
        else:  # Image sequence
            self.images = sorted(glob(os.path.join(self.source, '*.jpg')))
            self.frame_idx = 0

    def read_frame(self):
        """Read and return the next frame."""
        if not self.source:  # Webcam
            ret, frame = self.cap.read()
            return frame if ret else None
        elif self.source.endswith(('avi', 'mp4')):  # Video
            ret, frame = self.cap.read()
            return cv2.resize(frame, (640, 480)) if ret else None
        else:  # Image sequence
            if self.frame_idx < len(self.images):
                frame = cv2.imread(self.images[self.frame_idx])
                self.frame_idx += 1
                return cv2.resize(frame, (640, 480))
            return None

    def adjust_brightness_contrast(self, frame, brightness=0, contrast=0):
        """Optimized brightness/contrast adjustment using LUT."""
        brightness = np.clip(brightness, -100, 100)
        contrast = np.clip(contrast, -100, 100)
        
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

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            
class ObjectTracker:
    def __init__(self, config_path, model_path):
        self.cfg = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.tracker = build_tracker(self.model)
        self.bbox = None
        self.tracking = False

    def _load_config(self, config_path):
        """Load model configuration."""
        cfg = ...
        cfg.merge_from_file(config_path)
        cfg.CUDA = torch.cuda.is_available()
        return cfg

    def _load_model(self, model_path):
        """Load PyTorch model with half-precision if CUDA is available."""
        model = ModelBuilder()
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(self.device)
        if self.cfg.CUDA:
            model = model.half()  # FP16 for faster inference
        return model

    def init_tracking(self, frame, bbox):
        """Initialize tracker with a bounding box."""
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.tracking = True

    def update_tracking(self, frame):
        """Update tracker and return new bounding box."""
        if not self.tracking:
            return None
        
        outputs = self.tracker.track(frame)
        if 'bbox' in outputs:
            self.bbox = list(map(int, outputs['bbox']))
            return self.bbox
        return None

    def reset(self):
        """Reset tracker state."""
        self.bbox = None
        self.tracking = False
