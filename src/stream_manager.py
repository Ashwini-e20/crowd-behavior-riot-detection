
# stream_manager.py

from collections import deque
from datetime import datetime
import threading
import time
import os
import cv2
from PIL import Image
import numpy as np

import config
import video_processor 

# --- Global State Management (No changes needed) ---
class GlobalStateManager:
# ... (GlobalStateManager class remains unchanged) ...
    """Manages the application's global, thread-safe state."""

    def __init__(self):
        self.state = {
            "is_streaming": False,
            "video_source": 'webcam', # 'webcam' or filepath
            "last_people_count": 0,
            "last_weapon_count": 0, # <<< ADD THIS NEW FIELD
            "last_known_status": "Awaiting Input",
            "events": [],
            "last_event_time": None # Used to ensure the sound only plays once per event block
        }
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            return self.state.get(key)

    def set(self, key, value):
        with self.lock:
            self.state[key] = value

    def update(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def reset_for_new_stream(self, video_source):
        with self.lock:
            self.state.update({
                "video_source": video_source,
                "is_streaming": True,
                "last_people_count": 0,
                "last_known_status": "Starting File Analysis" if video_source != 'webcam' else "Normal",
                "events": [], 
                "last_event_time": None
            })


global_state_manager = GlobalStateManager()

# --- Video Streamer ---
class VideoStreamer:
    """Encapsulates the video processing logic and buffers."""
    def __init__(self, models):
        self.frames_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
        self.scores_buffer = deque(maxlen=config.SEQUENCE_LENGTH, iterable=[0.0] * config.SEQUENCE_LENGTH)
        self.detections_buffer = deque(maxlen=config.SEQUENCE_LENGTH, iterable=[[]] * config.SEQUENCE_LENGTH)
        self.anomaly_counter = 0
        self.frame_index = 0
        self.prediction_made_at = -config.ANOMALY_INTERVAL
        self.peak_anomaly_frame_index = -1 
        self.peak_anomaly_score = 0.0
        self.models = models 
        self.last_weapon_count = 0 # NEW: Track weapon count

    def process_frame(self, frame):
        """Processes a single frame to detect anomalies and update the UI."""
        
        self.frame_index += 1
        current_status = global_state_manager.get("last_known_status")

        # 1. Object Detection (YOLO) - MODIFIED CALL
        # The detect_people_and_weapons function now applies IoU filtering to all_detections
        last_people_count, all_detections, last_weapon_count = video_processor.detect_people_and_weapons(
            frame, self.models["yolo_people"], self.models["yolo_weapon"]
        )
        self.detections_buffer.append(all_detections) 
        self.last_weapon_count = last_weapon_count # Store the weapon count for potential use

        # 2. Frame/Feature Preparation
        processed_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frames_buffer.append(processed_frame_rgb)
        
        # --- ANOMALY PREDICTION: THROTTLED (Logic remains the same) ---
        can_predict = (self.frame_index - self.prediction_made_at) >= config.ANOMALY_INTERVAL

        if can_predict and len(self.frames_buffer) == config.SEQUENCE_LENGTH:
            
            # --- Perform Expensive Prediction ---
            resized_frames = [cv2.resize(f, (config.IMAGE_SIZE, config.IMAGE_SIZE)) for f in self.frames_buffer]
            features = video_processor.extract_features(resized_frames, self.models["efficientnet"])
            prediction, confidence = video_processor.predict_anomaly(features, self.models["gru"])
            
            self.prediction_made_at = self.frame_index 
            # --- End Expensive Prediction ---

            anomaly_score = confidence if prediction == 'Anomaly' else 1 - confidence
            self.scores_buffer.append(anomaly_score)
            
            # --- Update Peak Anomaly Tracking for accurate frame saving ---
            if anomaly_score > self.peak_anomaly_score:
                 self.peak_anomaly_score = anomaly_score
                 self.peak_anomaly_frame_index = config.SEQUENCE_LENGTH - 1 
            
            # --- State Transition Logic ---
            if prediction == 'Anomaly' and confidence > config.CONFIDENCE_THRESHOLD:
                self.anomaly_counter += 1
            else:
                self.anomaly_counter = 0
                if self.anomaly_counter == 0:
                     self.peak_anomaly_score = 0.0
                     self.peak_anomaly_frame_index = -1

            if self.anomaly_counter >= config.PERSISTENCE_THRESHOLD:
                new_status = "ANOMALY CONFIRMED"
                
                if current_status != "ANOMALY CONFIRMED":
                    # --- Event logging and saving (MODIFIED) ---
                    event_time = datetime.now()
                    event_id = event_time.strftime("%Y%m%d_%H%M%S")
                    
                    frame_to_save_index = self.peak_anomaly_frame_index if self.peak_anomaly_frame_index != -1 else config.SEQUENCE_LENGTH - 1
                    
                    peak_score = self.scores_buffer[frame_to_save_index]
                    trigger_frame_original = self.frames_buffer[frame_to_save_index]
                    # peak_detections now contains the *filtered* list of detections
                    peak_detections = self.detections_buffer[frame_to_save_index]
                    
                    frame_filename = f"frame_{event_id}.jpg"
                    clip_filename = f"clip_{event_id}.mp4"
                    
                    frame_path_full = os.path.join(config.TRIGGER_FRAMES_DIR, frame_filename)
                    clip_path_full = os.path.join(config.EVENT_CLIPS_DIR, clip_filename)

                    # CORRECT FIX: Pass ANOMALY_BOX_COLOR to draw boxes (people are red, weapon is blue)
                    trigger_frame_with_boxes = video_processor.draw_boxes_on_frame(
                        trigger_frame_original, peak_detections, config.ANOMALY_BOX_COLOR
                    )
                    Image.fromarray(trigger_frame_with_boxes).save(frame_path_full)
                    
                    threading.Thread(target=video_processor.save_event_clip, 
                                     args=(list(self.frames_buffer), clip_path_full)).start()

                    # MODIFIED: Event data counts are now derived from the *filtered* peak_detections
                    weapon_count_at_peak = len([d for d in peak_detections if d[-1] == 'weapon'])
                    people_count_at_peak = len([d for d in peak_detections if d[-1] == 'person'])
                    
                    event_data = {
                        "id": event_id, "timestamp": event_time.strftime("%I:%M:%S %p, %d-%b-%Y"),
                        "confidence": f"{peak_score:.2%}", 
                        "people_detected": people_count_at_peak,
                        "weapons_detected": weapon_count_at_peak, # NEW
                        "frame_file": frame_filename, 
                        "clip_file": clip_filename,
                    }
                    
                    with global_state_manager.lock:
                        global_state_manager.state["events"].append(event_data)
                        global_state_manager.state["last_event_time"] = event_time
            
            else:
                new_status = "Normal" if self.anomaly_counter == 0 else "Potential Anomaly"
        else:
             new_status = current_status


        # 4. Update Global State
        # last_people_count is correctly updated by video_processor.detect_people_and_weapons
        global_state_manager.update(
            last_known_status=new_status,
            last_people_count=last_people_count,
            last_weapon_count=last_weapon_count # NEW: Update weapon count in global state (if needed)
        )
        
        # 5. Draw Boxes for Display
        box_color = config.ANOMALY_BOX_COLOR if new_status == "ANOMALY CONFIRMED" else config.NORMAL_BOX_COLOR
        # all_detections passed here is the live, filtered frame's detections
        display_frame = video_processor.draw_boxes_on_frame(frame, all_detections, box_color)
        
        if new_status == "ANOMALY CONFIRMED":
            cv2.rectangle(display_frame, (0, 0), (frame.shape[1], frame.shape[0]), config.ANOMALY_BOX_COLOR, 10)
        
        return display_frame