# app_flask.py
from flask import Flask, Response, render_template, request, jsonify, url_for, send_from_directory
import cv2
import video_processor
import stream_manager # Import the new manager
import config
import tempfile
import os
import json
import threading
import time
from werkzeug.utils import secure_filename 
import shutil 
import math

# --- Setup ---
app = Flask(__name__)
app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.secret_key = 'super_secret_key_change_me' 

models = video_processor.load_all_models() 
# Instantiate the stream manager and streamer
global_state = stream_manager.global_state_manager
streamer = stream_manager.VideoStreamer(models)

# --- Utility Functions ---

def delayed_cleanup(filepath):
    """Attempt to delete the file after a short delay."""
    time.sleep(1) 
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Cleanup: Successfully deleted temporary file: {filepath}")
        except Exception as e:
            print(f"Cleanup Error: Could not delete {filepath} after delay. Error: {e}")

# --- Video Stream Generator (Uses global_state.get()) ---
def generate_frames():
    """Reads from webcam or file based on global_state['video_source']."""
    source = 0 if global_state.get("video_source") == 'webcam' else global_state.get("video_source")
    temp_file_path = global_state.get("video_source") if global_state.get("video_source") != 'webcam' else None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        global_state.update(is_streaming=False, last_known_status="Source Error")
        cap.release()
        return

    # Re-initialize the streamer state for a new run
    streamer.__init__(models) 

    while global_state.get("is_streaming"):
        ret, frame = cap.read()
        
        if not ret:
            if temp_file_path:
                global_state.update(is_streaming=False, last_known_status="File Analysis Complete")
            break
        
        processed_frame = streamer.process_frame(frame)
        
        # --- FRAME COMPRESSION ---
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    if global_state.get("is_streaming"): # Only update if not already set by an error
         global_state.update(
            is_streaming=False, 
            last_known_status="Stream Stopped" if global_state.get("video_source") == 'webcam' else "Analysis Finished"
        )

    # --- INITIATE DELAYED CLEANUP ---
    if temp_file_path:
        threading.Thread(target=delayed_cleanup, args=(temp_file_path,)).start()

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Renders the main dashboard page."""
    alert_sound_url = url_for('static', filename='alert.wav')
    return render_template('index.html', alert_sound_url=alert_sound_url)

@app.route('/log_files/<type>/<filename>')
def log_files(type, filename):
    """Serves saved files (clips or frames) securely from the log directories."""
    directory = config.TRIGGER_FRAMES_DIR if type == 'frame' else config.EVENT_CLIPS_DIR
    if type not in ['frame', 'clip']:
        return "Invalid file type", 404
    return send_from_directory(directory, filename)


@app.route('/video_feed')
def video_feed():
    """Serves the live video stream using multipart/x-mixed-replace."""
    if global_state.get("is_streaming"):
        return Response(generate_frames(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("", status=204) 

@app.route('/toggle_stream', methods=['POST'])
def toggle_stream():
    """Starts or stops the webcam stream based on the request data."""
    action = request.json.get('action')
    if action == 'start' and not global_state.get("is_streaming"):
        global_state.reset_for_new_stream('webcam')
        return jsonify({"status": "Stream started", "is_streaming": True})
    
    elif action == 'stop' and global_state.get("is_streaming"):
        global_state.update(is_streaming=False, last_known_status="Stream Stopped")
        return jsonify({"status": "Stream stopped", "is_streaming": False})

    return jsonify({"status": "Invalid action or already in state", "is_streaming": global_state.get("is_streaming")})


@app.route('/analytics_data')
def analytics_data():
    """Returns real-time analytics data and recent events as JSON."""
    
    current_state = global_state.state # Access the dict directly for a snapshot
    last_event_time_str = current_state["last_event_time"].isoformat() if current_state["last_event_time"] else None
    
    return jsonify({
        'people_count': current_state["last_people_count"],
        'status': current_state["last_known_status"],
        'is_streaming': current_state["is_streaming"],
        'video_source': current_state["video_source"],
        'events': current_state["events"][-5:],
        'last_event_time': last_event_time_str
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handles video file uploads and starts the analysis stream."""
    if 'video_file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400
    
    file = request.files['video_file']
    
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file:
        if global_state.get("is_streaming"):
             return jsonify({"message": "Cannot upload while a stream is active. Stop current stream first."}), 409
        
        temp_file_name = secure_filename(file.filename)
        # Use tempfile to ensure a unique and secure temporary file path
        # os.path.splitext is used to preserve the file extension for OpenCV
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(temp_file_name)[1])
        file.save(temp_file.name)
        
        # Reset state and start streaming from the file
        global_state.reset_for_new_stream(temp_file.name)
        
        return jsonify({
            "message": f"File '{file.filename}' uploaded successfully. Analysis started.", 
            "filepath": temp_file.name,
            "is_streaming": True
        }), 200

if __name__ == '__main__':
    # Initialize static directory and copy alert.wav
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    if not os.path.exists(static_path):
        os.makedirs(static_path)

    alert_source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alert.wav')
    alert_dest = os.path.join(static_path, 'alert.wav')
    if os.path.exists(alert_source) and not os.path.exists(alert_dest):
        shutil.copy2(alert_source, alert_dest)

    app.run(host='0.0.0.0', port=5000, threaded=True)