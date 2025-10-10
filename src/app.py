import streamlit as st
import cv2
from collections import deque
import tempfile
import os
from PIL import Image

import config
import video_processor

# ===================================================================
# STREAMLIT UI SETUP
# ===================================================================

st.set_page_config(layout="wide", page_title="Real-Time Anomaly Detection")
st.title("🚨 AI-Based Surveillance System")
st.markdown("<style>audio { display:none; }</style>", unsafe_allow_html=True)

@st.cache_resource
def get_models():
    """Load all models once and cache them."""
    return video_processor.load_all_models()

models = get_models()

# --- Session State Initialization ---
if 'anomaly_counter' not in st.session_state: st.session_state.anomaly_counter = 0
if 'system_status' not in st.session_state: st.session_state.system_status = "Normal"
if 'alert_played' not in st.session_state: st.session_state.alert_played = False
if 'is_alert_active' not in st.session_state: st.session_state.is_alert_active = False
if 'trigger_frame_saved' not in st.session_state: st.session_state.trigger_frame_saved = False
if 'trigger_frame_path' not in st.session_state: st.session_state.trigger_frame_path = ""

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
    # Placeholder for the new on-demand explainability report
    explain_placeholder = st.empty()

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Reset state for the new video
    st.session_state.anomaly_counter = 0
    st.session_state.system_status = "Normal"
    st.session_state.alert_played = False
    st.session_state.is_alert_active = False
    st.session_state.trigger_frame_saved = False
    st.session_state.trigger_frame_path = ""
    audio_placeholder.empty()
    explain_placeholder.empty()
    
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    frame_count, last_people_count, last_detections = 0, 0, []
    frames_buffer = deque(maxlen=config.SEQUENCE_LENGTH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_count % config.YOLO_INTERVAL == 0:
            last_people_count, last_detections = video_processor.detect_people(frame, models["yolo"])
        
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame_resized = cv2.resize(processed_frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        frames_buffer.append(processed_frame_resized)

        if len(frames_buffer) == config.SEQUENCE_LENGTH:
            features = video_processor.extract_features(list(frames_buffer), models["efficientnet"])
            prediction, confidence = video_processor.predict_anomaly(features, models["gru"])
            
            if prediction == 'Anomaly' and confidence > config.CONFIDENCE_THRESHOLD:
                st.session_state.anomaly_counter += 1
            else:
                st.session_state.anomaly_counter = 0

            if st.session_state.anomaly_counter >= config.PERSISTENCE_THRESHOLD:
                st.session_state.system_status = "ANOMALY CONFIRMED"
                st.session_state.is_alert_active = True
                if not st.session_state.alert_played:
                    audio_placeholder.audio(config.ALERT_SOUND_PATH, autoplay=True)
                    st.session_state.alert_played = True
                
                # --- NEW: Save frame and INSTANTLY show the explanation option ---
                if not st.session_state.trigger_frame_saved:
                    # Save the frame that triggered the persistent alert
                    trigger_frame = list(frames_buffer)[-1]
                    img_to_save = Image.fromarray(trigger_frame)
                    st.session_state.trigger_frame_path = os.path.join(tempfile.gettempdir(), "trigger_frame.jpg")
                    img_to_save.save(st.session_state.trigger_frame_path)
                    st.session_state.trigger_frame_saved = True

                    # Use the placeholder to create the expander on-demand
                    with explain_placeholder.container():
                        with st.expander("🔬 Explainability Report", expanded=True):
                            st.info("An anomaly was confirmed. Click below to see why.")
                            if st.button("Show Explanation"):
                                with st.spinner("Generating Grad-CAM heatmap..."):
                                    grad_cam_image = video_processor.generate_grad_cam_on_saved_frame(
                                        st.session_state.trigger_frame_path, models
                                    )
                                    st.image(grad_cam_image, caption="Why the model flagged an anomaly.")
            
            else: # If not a confirmed anomaly
                st.session_state.is_alert_active = False
                st.session_state.system_status = "Normal" if st.session_state.anomaly_counter == 0 else "Potential Anomaly"
                if st.session_state.alert_played: # Re-arm alert
                    st.session_state.alert_played = False
                    st.session_state.trigger_frame_saved = False
                    audio_placeholder.empty()
                    explain_placeholder.empty() # Clear the explanation UI
        
        # --- Update UI ---
        people_count_widget.metric("People Detected", last_people_count)
        if st.session_state.system_status == "ANOMALY CONFIRMED": status_widget.error(st.session_state.system_status)
        elif st.session_state.system_status == "Potential Anomaly": status_widget.warning(st.session_state.system_status)
        else: status_widget.success(st.session_state.system_status)

        if st.session_state.is_alert_active:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        for x1, y1, x2, y2 in last_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        frame_count += 1
    
    cap.release()
    st.success("Video processing complete.")
else:
    st.info("Please upload a video to begin analysis.")

