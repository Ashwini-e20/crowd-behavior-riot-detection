# stream_manager.py

from collections import deque
from datetime import datetime
import threading
import time
import os
import cv2
from PIL import Image
import numpy as np
from playsound import playsound   # ✅ Added for alert sound playback

import config
import video_processor 


# --- ALERT SOUND HELPER ---
def play_alert_sound():
    """Play alert sound asynchronously to avoid blocking main thread."""
    def _play():
        try:
            playsound(os.path.join("static", "alert.wav"))
        except Exception as e:
            print(f"[Warning] Could not play alert sound: {e}")
    threading.Thread(target=_play, daemon=True).start()


# --- Global State Management ---
class GlobalStateManager:
    """Manages the application's global, thread-safe state."""
    def __init__(self):
        self.state = {
            "is_streaming": False,
            "video_source": 'webcam',  # 'webcam' or filepath
            "last_people_count": 0,
            "last_weapon_count": 0,
            "last_known_status": "Awaiting Input",
            "events": [],
            "last_event_time": None  # Used to ensure sound plays once per event
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
                "last_weapon_count": 0,
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
        self.last_weapon_count = 0

    def process_frame(self, frame):
        """Processes a single frame to detect anomalies and update the UI."""
        self.frame_index += 1
        current_status = global_state_manager.get("last_known_status")

        # --- 1. Object Detection ---
        last_people_count, all_detections, last_weapon_count = video_processor.detect_people_and_weapons(
            frame, self.models["yolo_people"], self.models["yolo_weapon"]
        )
        self.detections_buffer.append(all_detections)
        self.last_weapon_count = last_weapon_count

        # --- 2. Frame Preparation ---
        processed_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frames_buffer.append(processed_frame_rgb)

        can_predict = (self.frame_index - self.prediction_made_at) >= config.ANOMALY_INTERVAL
        if can_predict and len(self.frames_buffer) == config.SEQUENCE_LENGTH:
            resized_frames = [cv2.resize(f, (config.IMAGE_SIZE, config.IMAGE_SIZE)) for f in self.frames_buffer]
            features = video_processor.extract_features(resized_frames, self.models["efficientnet"])
            prediction, confidence = video_processor.predict_anomaly(features, self.models["gru"])
            self.prediction_made_at = self.frame_index

            anomaly_score = confidence if prediction == 'Anomaly' else 1 - confidence
            self.scores_buffer.append(anomaly_score)

            if anomaly_score > self.peak_anomaly_score:
                self.peak_anomaly_score = anomaly_score
                self.peak_anomaly_frame_index = config.SEQUENCE_LENGTH - 1

            if prediction == 'Anomaly' and confidence > config.CONFIDENCE_THRESHOLD:
                self.anomaly_counter += 1
            else:
                self.anomaly_counter = 0
                self.peak_anomaly_score = 0.0
                self.peak_anomaly_frame_index = -1

            if self.anomaly_counter >= config.PERSISTENCE_THRESHOLD:
                new_status = "ANOMALY CONFIRMED"
                if current_status != "ANOMALY CONFIRMED":
                    event_time = datetime.now()
                    event_id = event_time.strftime("%Y%m%d_%H%M%S")

                    frame_to_save_index = self.peak_anomaly_frame_index if self.peak_anomaly_frame_index != -1 else config.SEQUENCE_LENGTH - 1
                    peak_score = self.scores_buffer[frame_to_save_index]
                    trigger_frame_original = self.frames_buffer[frame_to_save_index]
                    peak_detections = self.detections_buffer[frame_to_save_index]

                    frame_filename = f"frame_{event_id}.jpg"
                    clip_filename = f"clip_{event_id}.mp4"
                    frame_path_full = os.path.join(config.TRIGGER_FRAMES_DIR, frame_filename)
                    clip_path_full = os.path.join(config.EVENT_CLIPS_DIR, clip_filename)

                    trigger_frame_with_boxes = video_processor.draw_boxes_on_frame(
                        trigger_frame_original, peak_detections, config.ANOMALY_BOX_COLOR
                    )
                    Image.fromarray(trigger_frame_with_boxes).save(frame_path_full)
                    threading.Thread(target=video_processor.save_event_clip,
                                     args=(list(self.frames_buffer), clip_path_full)).start()

                    weapon_count_at_peak = len([d for d in peak_detections if d[-1] == 'weapon'])
                    people_count_at_peak = len([d for d in peak_detections if d[-1] == 'person'])
                    event_data = {
                        "id": event_id,
                        "timestamp": event_time.strftime("%I:%M:%S %p, %d-%b-%Y"),
                        "confidence": f"{peak_score:.2%}",
                        "people_detected": people_count_at_peak,
                        "weapons_detected": weapon_count_at_peak,
                        "frame_file": frame_filename,
                        "clip_file": clip_filename,
                    }
                    with global_state_manager.lock:
                        global_state_manager.state["events"].append(event_data)
                        global_state_manager.state["last_event_time"] = datetime.now()
                    play_alert_sound()

            else:
                new_status = "Normal" if self.anomaly_counter == 0 else "Potential Anomaly"
        else:
            new_status = current_status

        # PLAY ALERT SOUND IF WEAPON DETECTED (COOLDOWN)
        if last_weapon_count > 0:
            last_played = global_state_manager.get("last_event_time")
            now = datetime.now()
            if not last_played or (now - last_played).total_seconds() > 5:
                play_alert_sound()
                global_state_manager.set("last_event_time", now)


        # --- 4. Update Global State ---
        global_state_manager.update(
            last_known_status=new_status,
            last_people_count=last_people_count,
            last_weapon_count=last_weapon_count
        )

        # --- 5. Draw Boxes ---
        box_color = config.ANOMALY_BOX_COLOR if new_status == "ANOMALY CONFIRMED" else config.NORMAL_BOX_COLOR
        display_frame = video_processor.draw_boxes_on_frame(frame, all_detections, box_color)
        if new_status == "ANOMALY CONFIRMED":
            cv2.rectangle(display_frame, (0, 0), (frame.shape[1], frame.shape[0]), config.ANOMALY_BOX_COLOR, 10)

        return display_frame
