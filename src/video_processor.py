# video_processor.py

import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from ultralytics import YOLO
import cv2 

from models import GRUModel
import config

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- IoU Calculation (Utility function - not used for cross-class suppression in this version) ---
def calculate_iou(boxA, boxB):
    """Calculates Intersection Over Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    
    # Unpack coordinates (only need the first 4 for coordinates)
    x1A, y1A, x2A, y2A = boxA[:4]
    x1B, y1B, x2B, y2B = boxB[:4]

    # Determine the coordinates of the intersection rectangle
    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction boxes
    boxAArea = (x2A - x1A) * (y2A - y1A)
    boxBArea = (x2B - x1B) * (y2B - y1B)

    # Compute the union area, and add a small epsilon (1e-6) to prevent division by zero
    unionArea = boxAArea + boxBArea - interArea
    iou = interArea / float(unionArea + 1e-6)
    
    return iou


# --- CORE MODEL LOADING ---
def load_all_models():
    """Loads all detection and anomaly models onto the specified device."""
    print("Loading models...")
    
    # 1. YOLO Models
    yolo_people_model = YOLO(config.YOLO_MODEL_PATH)
    yolo_weapon_model = YOLO(config.WEAPON_YOLO_MODEL_PATH)
    
    # 2. EfficientNet Feature Extractor
    efficientnet = EfficientNet.from_pretrained(config.EFFICIENTNET_MODEL_NAME)
    # Replace the final classification layer with identity to get features
    efficientnet._fc = torch.nn.Identity()
    efficientnet = efficientnet.to(config.DEVICE)
    efficientnet.eval()
    
    # 3. GRU Anomaly Predictor
    gru_model = GRUModel().to(config.DEVICE)
    gru_model.load_state_dict(torch.load(config.GRU_MODEL_PATH, map_location=config.DEVICE))
    gru_model.eval()
    
    print("Models loaded successfully.")
    return {
        "yolo_people": yolo_people_model, 
        "yolo_weapon": yolo_weapon_model, 
        "efficientnet": efficientnet, 
        "gru": gru_model
    }


# --- DETECTION LOGIC (No Suppression for accuracy) ---
def detect_people_and_weapons(frame, yolo_people_model, yolo_weapon_model):
    """
    Runs both YOLO models, combines detections, and returns counts.
    Cross-class suppression (person/weapon overlap) is intentionally omitted
    to ensure a weapon detected on a person is never filtered out.
    """
    
    # --- 1. Run People Detector (COCO trained) ---
    people_results = yolo_people_model(frame, verbose=False)
    # Filter for class 0 (person)
    person_detections = [b for b in people_results[0].boxes if b.cls == config.PEOPLE_CLASS_ID]
    people_boxes_raw = [list(map(int, d.xyxy[0])) for d in person_detections]
    people_detections_labeled = [(*box, 'person') for box in people_boxes_raw]
    
    # --- 2. Run Weapon Detector (Custom trained) ---
    weapon_results = yolo_weapon_model(frame, verbose=False)
    # Filter for class 1 (weapon - based on config)
    weapon_detections = [b for b in weapon_results[0].boxes if b.cls == config.WEAPON_CLASS_ID]
    weapon_boxes_raw = [list(map(int, d.xyxy[0])) for d in weapon_detections]

    # --- 3. Suppression Step (INTENTIONALLY OMITTED/REMOVED) ---
    # The fix is to skip any code here that filters weapon boxes based on IoU with person boxes.
    
    final_weapon_detections = [(*box, 'weapon') for box in weapon_boxes_raw]

    # --- 4. Combine and Return ---
    # All people detections and all detected weapons are combined
    all_detections = people_detections_labeled + final_weapon_detections
    
    last_people_count = len(people_boxes_raw)
    last_weapon_count = len(final_weapon_detections)
    
    return last_people_count, all_detections, last_weapon_count


# --- FEATURE EXTRACTION ---
def extract_features(frames, efficientnet_model):
    """Extracts features from a sequence of frames using EfficientNet."""
    with torch.no_grad():
        # Convert list of PIL Images/Arrays to a single tensor batch and normalize
        batch = torch.stack([_transform(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) for f in frames]).to(config.DEVICE)
        features = efficientnet_model(batch)
    return features.cpu().numpy()


# --- ANOMALY PREDICTION ---
def predict_anomaly(features, gru_model):
    """Predicts anomaly class and confidence using the GRU model."""
    # Convert features to a sequence tensor (Batch_size=1, Sequence_Length, Feature_Dim)
    sequence_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        output = gru_model(sequence_tensor)
        probabilities = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probabilities, 1)
        
        # Get labels from config
        prediction = config.CLASS_NAMES[pred_idx.item()]
        confidence = conf.item()
        
    return prediction, confidence


# --- VISUALIZATION AND LOGGING UTILITIES ---
def save_event_clip(frame_buffer, output_path, fps=10):
    """Saves a deque of frames as a video clip."""
    if not frame_buffer:
        return
    # Use the shape of the first frame (assuming all are the same size)
    height, width, _ = frame_buffer[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frame_buffer:
        # Frames in buffer are stored in RGB (PIL/Numpy format); convert back to BGR for OpenCV
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def draw_boxes_on_frame(frame, detections, anomaly_status_color, thickness=2):
    """
    Draws bounding boxes. Weapons are drawn in blue (WEAPON_BOX_COLOR), 
    while people are drawn in the current status color (NORMAL_BOX_COLOR or ANOMALY_BOX_COLOR).
    """
    # The input frame is expected to be BGR (from OpenCV read), copy it for drawing
    frame_copy = frame.copy()
    
    for x1, y1, x2, y2, det_type in detections:
        # Determine color based on detection type
        if det_type == 'weapon':
            # Highlight weapon distinctly in blue
            color = config.WEAPON_BOX_COLOR 
        elif det_type == 'person':
            # Use the status color passed (e.g., Red during confirmed anomaly)
            color = anomaly_status_color 
        else:
            # Fallback color, should not happen if detection types are correct
            color = anomaly_status_color 
            
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
        
        label = det_type.upper()
        # Draw label text above the box
        cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return frame_copy