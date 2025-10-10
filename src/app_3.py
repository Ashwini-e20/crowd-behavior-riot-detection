# app_3.py

import streamlit as st
import cv2
from collections import deque
import tempfile
import os
import numpy as np
from datetime import datetime
from PIL import Image

# Custom module imports
import config
import video_processor

# ===================================================================
# STREAMLIT APP
# ===================================================================

st.set_page_config(layout="wide", page_title="AI Surveillance System")
st.title("AI-Based Surveillance System")

# Hides the audio player UI
st.markdown("<style>audio { display:none; }</style>", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_models_cached():
    """Load all models once and cache them."""
    return video_processor.load_all_models()

models = load_models_cached()

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes or resets the session state for a monitoring session."""
    st.session_state.events = []
    st.session_state.run_webcam = False
    st.session_state.last_event_count = 0
    # Buffers for processing
    st.session_state.frames_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    st.session_state.scores_buffer = deque(maxlen=config.SEQUENCE_LENGTH, iterable=[0.0]*config.SEQUENCE_LENGTH)
    st.session_state.detections_buffer = deque(maxlen=config.SEQUENCE_LENGTH, iterable=[[]]*config.SEQUENCE_LENGTH)
    # State tracking variables
    st.session_state.anomaly_counter = 0
    st.session_state.last_known_status = "Normal"

if 'events' not in st.session_state:
    initialize_session_state()

# --- UI Layout ---
col1, col2 = st.columns([3, 1])
with col1:
    video_placeholder = st.empty()
    audio_placeholder = st.empty()
with col2:
    st.subheader("Live Analytics")
    people_count_widget = st.empty()
    st.subheader("System Status")
    status_widget = st.empty()
    st.subheader("Event Log")
    event_log_placeholder = st.empty()

# --- Main Application Logic ---

def process_frame(frame):
    """
    Processes a single video frame to detect anomalies and update the UI.
    """
    # --- Object Detection ---
    last_people_count, last_detections = video_processor.detect_people(frame, models["yolo"])
    st.session_state.detections_buffer.append(last_detections)

    # --- Frame Preparation ---
    processed_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.frames_buffer.append(processed_frame_rgb)
    
    status = st.session_state.last_known_status

    # --- Anomaly Prediction ---
    if len(st.session_state.frames_buffer) == config.SEQUENCE_LENGTH:
        resized_frames = [cv2.resize(f, (config.IMAGE_SIZE, config.IMAGE_SIZE)) for f in st.session_state.frames_buffer]
        features = video_processor.extract_features(resized_frames, models["efficientnet"])
        prediction, confidence = video_processor.predict_anomaly(features, models["gru"])

        anomaly_score = confidence if prediction == 'Anomaly' else 1 - confidence
        st.session_state.scores_buffer.append(anomaly_score)

        if prediction == 'Anomaly' and confidence > config.CONFIDENCE_THRESHOLD:
            st.session_state.anomaly_counter += 1
        else:
            st.session_state.anomaly_counter = 0

        # Event Trigger Logic using State Transition
        if st.session_state.anomaly_counter >= config.PERSISTENCE_THRESHOLD:
            status = "ANOMALY CONFIRMED"
            if st.session_state.last_known_status != "ANOMALY CONFIRMED":
                event_time = datetime.now()
                event_id = event_time.strftime("%Y%m%d_%H%M%S")
                
                peak_index = np.argmax(list(st.session_state.scores_buffer))
                peak_score = st.session_state.scores_buffer[peak_index]
                trigger_frame_original = st.session_state.frames_buffer[peak_index]
                peak_detections = st.session_state.detections_buffer[peak_index]

                trigger_frame_with_boxes = video_processor.draw_boxes_on_frame(
                    trigger_frame_original, peak_detections, config.ANOMALY_BOX_COLOR
                )
                
                frame_path = os.path.join(config.TRIGGER_FRAMES_DIR, f"frame_{event_id}.jpg")
                clip_path = os.path.join(config.EVENT_CLIPS_DIR, f"clip_{event_id}.mp4")
                
                Image.fromarray(trigger_frame_with_boxes).save(frame_path)
                video_processor.save_event_clip(list(st.session_state.frames_buffer), clip_path)

                event_data = {
                    "id": event_id, "timestamp": event_time.strftime("%I:%M:%S %p, %d-%b-%Y"),
                    "confidence": f"{peak_score:.2%}", "people_detected": len(peak_detections),
                    "frame_path": frame_path, "clip_path": clip_path,
                }
                st.session_state.events.append(event_data)
        else:
            status = "Normal" if st.session_state.anomaly_counter == 0 else "Potential Anomaly"
        
        st.session_state.last_known_status = status

    # --- UI Rendering ---
    box_color = config.ANOMALY_BOX_COLOR if status == "ANOMALY CONFIRMED" else config.NORMAL_BOX_COLOR
    display_frame = video_processor.draw_boxes_on_frame(frame, last_detections, box_color)
    
    if status == "ANOMALY CONFIRMED":
        cv2.rectangle(display_frame, (0, 0), (frame.shape[1], frame.shape[0]), config.ANOMALY_BOX_COLOR, 10)

    # --- MODIFIED: Replaced deprecated argument ---
    video_placeholder.image(display_frame, channels="BGR", width='stretch')
    
    people_count_widget.metric("People Detected", last_people_count)

    if status == "ANOMALY CONFIRMED":
        if len(st.session_state.events) > st.session_state.last_event_count:
            audio_placeholder.audio(config.ALERT_SOUND_PATH, autoplay=True)
            st.session_state.last_event_count = len(st.session_state.events)
        status_widget.error(status)
    elif status == "Potential Anomaly":
        status_widget.warning(status)
    else:
        status_widget.success(status)

    with event_log_placeholder.container():
        if not st.session_state.events:
            st.info("No anomalous events detected yet.")
        else:
            for event in reversed(st.session_state.events):
                with st.expander(f"Anomaly at {event['timestamp']}", expanded=True):
                    st.image(event['frame_path'], caption=f"Trigger Frame (Event ID: {event['id']})")
                    c1, c2 = st.columns(2)
                    c1.metric("Confidence", event['confidence'])
                    c2.metric("People Detected", event['people_detected'])
                    if os.path.exists(event['clip_path']):
                        with open(event['clip_path'], 'rb') as video_file:
                            st.video(video_file.read())
                    else:
                        st.warning("Event clip is being processed...")

# --- UI Input Handling ---
st.sidebar.title("Input Source")
input_source = st.sidebar.radio("Choose an input source:", ('Upload Video', 'Live Webcam Feed'))

# Set initial UI state when no input is processed yet
if 'start_button_pressed' not in st.session_state:
    st.session_state.start_button_pressed = False

is_processing = False
if input_source == 'Upload Video':
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        is_processing = True
elif input_source == 'Live Webcam Feed':
    if st.sidebar.button("Start Webcam", key="start_webcam"):
        st.session_state.run_webcam = True
        is_processing = True
    if st.sidebar.button("Stop Webcam", key="stop_webcam"):
        st.session_state.run_webcam = False
    if st.session_state.run_webcam:
        is_processing = True

if not is_processing:
    people_count_widget.metric("People Detected", 0)
    status_widget.info("Awaiting Input...")
    video_placeholder.info("Select an input source and start the analysis.", icon="📹")

# --- Main Execution Block ---
if input_source == 'Upload Video' and uploaded_file:
    initialize_session_state()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        process_frame(frame)
    cap.release()
    st.success("Video processing complete.")

elif input_source == 'Live Webcam Feed' and st.session_state.run_webcam:
    # A new session state should be initialized only when 'Start' is clicked.
    if 'cam_initialized' not in st.session_state or not st.session_state.cam_initialized:
        initialize_session_state()
        st.session_state.cam_initialized = True

    cap = cv2.VideoCapture(0)
    st.sidebar.info("Webcam feed is live...")
    while st.session_state.run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            st.session_state.run_webcam = False
            break
        process_frame(frame)
    cap.release()
    st.session_state.cam_initialized = False # Reset for next start