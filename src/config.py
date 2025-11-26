# config.py

import torch
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# --- File Paths ---
GRU_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gru", "best_gru_model.pth")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo", "yolov8n.pt")
# NEW: Placeholder for Weapon Detection Model Path
WEAPON_YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo", "best(1).pt") # << PLACEHOLDER
ALERT_SOUND_PATH = os.path.join(PROJECT_ROOT, "alert.wav")

# --- Event Logging & Output Configuration ---
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
EVENT_CLIPS_DIR = os.path.join(LOGS_DIR, "event_clips") # For saving short video clips
TRIGGER_FRAMES_DIR = os.path.join(LOGS_DIR, "trigger_frames") # For saving still images
EVENT_LOG_PATH = os.path.join(LOGS_DIR, "anomaly_log.csv") # For a text-based log

# --- Model & Processing Parameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 100
IMAGE_SIZE = 224
CLASS_NAMES = ['Normal', 'Anomaly']
EFFICIENTNET_MODEL_NAME = 'efficientnet-b0'

# --- Real-Time Application Parameters ---
YOLO_INTERVAL = 5
ANOMALY_INTERVAL = 10
CONFIDENCE_THRESHOLD = 0.65
PERSISTENCE_THRESHOLD = 3
# NEW: Threshold for suppressing false positive weapon detections that overlap people
IOU_SUPPRESSION_THRESHOLD = 0.8

# --- MODIFIED: Bounding Box Colors ---
NORMAL_BOX_COLOR = (0, 255, 0) # Green
ANOMALY_BOX_COLOR = (0, 0, 255) # Red
WEAPON_BOX_COLOR = (255, 0, 0) # Blue for weapon detection (NEW)

# --- Detection Parameters ---
PEOPLE_CLASS_ID = 0 # YOLOv8n default for person
WEAPON_CLASS_ID = 0 # Assuming the weapon model uses class 1 for weapons (NEW)

# --- Create directories if they don't exist ---
os.makedirs(EVENT_CLIPS_DIR, exist_ok=True)
os.makedirs(TRIGGER_FRAMES_DIR, exist_ok=True)