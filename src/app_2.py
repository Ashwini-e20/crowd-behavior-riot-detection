import streamlit as st
import cv2
from collections import deque
import tempfile
import os
import numpy as np

# Assuming your custom modules are in the same directory or in the python path
import config
import video_processor

# ===================================================================
# STREAMLIT UI SETUP
# ===================================================================

st.set_page_config(layout="wide", page_title="Real-Time Anomaly Detection")
st.title("🚨 AI-Based Surveillance System")
st.markdown("<style>audio { display:none; }</style>", unsafe_allow_html=True)

# --- Load Models (cached for efficiency) ---
@st.cache_resource
def load_models():
    """Loads all required models and caches them."""
    return video_processor.load_all_models()

models = load_models()

# --- Session State Initialization ---
def initialize_state():
    """Initializes or resets the session state variables."""
    st.session_state.frames_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    st.session_state.anomaly_counter = 0
    st.session_state.system_status = "Normal"
    st.session_state.alert_played = False
    st.session_state.is_alert_active = False
    st.session_state.trigger_sequence_saved = False
    st.session_state.trigger_sequence_path = ""
    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False

# Initialize state on first run
if 'system_status' not in st.session_state:
    initialize_state()

# --- UI Layout ---
col1, col2 = st.columns([3, 1])
with col1:
    video_placeholder = st.empty()
    audio_placeholder = st.empty()
with col2:
    st.subheader("📊 Live Analytics")
    people_count_widget = st.empty()
    st.subheader("📈 System Status")
    status_widget = st.empty()
    explain_placeholder = st.empty()

# ===================================================================
# MAIN PROCESSING LOGIC
# ===================================================================

def process_frame(frame, models):
    """
    Processes a single frame for anomaly detection and updates the UI.
    This function is shared between video file and webcam processing.
    """
    # --- People Detection (runs periodically) ---
    if st.session_state.get('frame_count', 0) % config.YOLO_INTERVAL == 0:
        st.session_state.last_people_count, st.session_state.last_detections = video_processor.detect_people(frame, models["yolo"])
    
    # --- Anomaly Detection ---
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame_resized = cv2.resize(processed_frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    st.session_state.frames_buffer.append(processed_frame_resized)

    if len(st.session_state.frames_buffer) == config.SEQUENCE_LENGTH:
        features = video_processor.extract_features(list(st.session_state.frames_buffer), models["efficientnet"])
        prediction, confidence = video_processor.predict_anomaly(features, models["gru"])
        
        # --- Anomaly Persistence Check ---
        if prediction == 'Anomaly' and confidence > config.CONFIDENCE_THRESHOLD:
            st.session_state.anomaly_counter += 1
        else:
            st.session_state.anomaly_counter = 0

        # --- Confirmed Anomaly Logic ---
        if st.session_state.anomaly_counter >= config.PERSISTENCE_THRESHOLD:
            st.session_state.system_status = "ANOMALY CONFIRMED"
            st.session_state.is_alert_active = True
            if not st.session_state.alert_played:
                audio_placeholder.audio(config.ALERT_SOUND_PATH, autoplay=True)
                st.session_state.alert_played = True
            
            # Save the frame sequence that triggered the anomaly for explainability
            if not st.session_state.trigger_sequence_saved:
                trigger_sequence = np.array(list(st.session_state.frames_buffer))
                st.session_state.trigger_sequence_path = os.path.join(tempfile.gettempdir(), "trigger_sequence.npy")
                np.save(st.session_state.trigger_sequence_path, trigger_sequence)
                st.session_state.trigger_sequence_saved = True

                with explain_placeholder.container():
                    with st.expander("🔬 Explainability Report", expanded=True):
                        st.info("Anomaly confirmed. Click below for an Integrated Gradients analysis.")
                        if st.button("Show Explanation"):
                            with st.spinner("Generating explanation... This may take a moment."):
                                explanation_fig = video_processor.generate_ig_explanation(
                                    st.session_state.trigger_sequence_path, models
                                )
                                st.pyplot(explanation_fig)
        
        # --- Normal/Potential Anomaly Logic ---
        else:
            st.session_state.is_alert_active = False
            st.session_state.system_status = "Normal" if st.session_state.anomaly_counter == 0 else "Potential Anomaly"
            if st.session_state.alert_played:
                st.session_state.alert_played = False
                st.session_state.trigger_sequence_saved = False
                audio_placeholder.empty()
                explain_placeholder.empty()
            
    # --- Update UI Widgets ---
    people_count_widget.metric("People Detected", st.session_state.get('last_people_count', 0))
    if st.session_state.system_status == "ANOMALY CONFIRMED":
        status_widget.error(st.session_state.system_status)
    elif st.session_state.system_status == "Potential Anomaly":
        status_widget.warning(st.session_state.system_status)
    else:
        status_widget.success(st.session_state.system_status)

    # --- Draw Bounding Boxes and Alert Borders on the Frame ---
    if st.session_state.is_alert_active:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
    for x1, y1, x2, y2 in st.session_state.get('last_detections', []):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    video_placeholder.image(frame, channels="BGR", use_container_width=True)
    st.session_state.frame_count += 1

# ===================================================================
# INPUT SOURCE SELECTION AND EXECUTION
# ===================================================================

input_source = st.radio("Choose Input Source", ('Upload Video', 'Webcam Feed'), horizontal=True)

if input_source == 'Upload Video':
    st.session_state.run_webcam = False # Ensure webcam is off
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        initialize_state() # Reset state for the new video
        st.session_state.frame_count = 0
        st.session_state.last_people_count = 0
        st.session_state.last_detections = []
        
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(frame, models)
        
        cap.release()
        st.success("Video processing complete.")
    else:
        st.info("Please upload a video to begin analysis.")

elif input_source == 'Webcam Feed':
    btn_cols = st.columns(2)
    if btn_cols[0].button("Start Webcam Feed", use_container_width=True):
        initialize_state() # Reset state for the new stream
        st.session_state.run_webcam = True
        st.session_state.frame_count = 0
        st.session_state.last_people_count = 0
        st.session_state.last_detections = []
        st.rerun()

    if btn_cols[1].button("Stop Webcam Feed", use_container_width=True):
        st.session_state.run_webcam = False
        st.rerun()

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Please check permissions and connections.")
        else:
            st.info("Webcam feed is live. Click 'Stop' to end.")
            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image from webcam.")
                    break
                process_frame(frame, models)
            cap.release()
    else:
        st.info("Click 'Start Webcam Feed' to begin live analysis.")