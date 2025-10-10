# app_flask.py
from flask import Flask, Response, render_template, request, jsonify, url_for, send_from_directory
import cv2
import video_processor
import stream_manager 
import config
import tempfile
import os
import threading
import time
from werkzeug.utils import secure_filename 
import shutil 


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
    """Reads from webcam (0), file, or IP URL based on global_state['video_source']."""
    
    source = global_state.get("video_source")
    
    # OpenCV uses 0 for default webcam, or the URL/filepath string
    cap_source = 0 if source == 'webcam' else source
    
    # Track file path for cleanup, ignore 'webcam' and network streams (which don't need cleanup)
    # File streams are expected to be local file paths (not starting with http/rtsp)
    temp_file_path = source if source != 'webcam' and not source.startswith('http') and not source.startswith('rtsp') else None

    # Try to open the video source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {cap_source}")
        # IMPORTANT: Set status to Source Error if opening the capture failed
        global_state.update(is_streaming=False, last_known_status="Source Error")
        cap.release()
        return

    # Re-initialize the streamer state for a new run
    streamer.__init__(models) 

    while global_state.get("is_streaming"):
        ret, frame = cap.read()
        
        if not ret:
            # If streaming from a file, this means the end of the file.
            if temp_file_path:
                global_state.update(is_streaming=False, last_known_status="File Analysis Complete")
            # If streaming from webcam/IP, this means the connection was lost.
            else:
                 global_state.update(is_streaming=False, last_known_status="Source Error")
            break
        
        # Guard against empty frames, common with unstable network streams
        if frame is None or frame.size == 0:
            continue
            
        processed_frame = streamer.process_frame(frame)
        
        # --- FRAME COMPRESSION ---
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    
    # Cleanup logic remains the same, ensures status is only set if not already set by error/completion
    if global_state.get("is_streaming"): 
         global_state.update(
             is_streaming=False, 
             last_known_status="Stream Stopped"
         )

    # --- INITIATE DELAYED CLEANUP ---
    if temp_file_path:
        threading.Thread(target=delayed_cleanup, args=(temp_file_path,)).start()

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Renders the main dashboard page."""
    alert_sound_url = url_for('static', filename='alert.wav')
    # MODIFIED: Pass a placeholder URL that the JS can use when the stream is inactive
    placeholder_url = url_for('static', filename='placeholder.jpeg')
    return render_template('index.html', alert_sound_url=alert_sound_url, placeholder_url=placeholder_url)

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
    """
    Starts or stops the stream. 
    """
    data = request.json
    action = data.get('action')
    # Get the source URL/ID from the request payload. Default to 'webcam' (index 0).
    source = data.get('source', 'webcam') 

    if action == 'start' and not global_state.get("is_streaming"):
        
        # Reset state and start streaming from the provided source
        global_state.reset_for_new_stream(source)
        
        # Determine the source for the display message
        if source == 'webcam':
            source_display = 'Webcam (Index 0)'
            initial_status = "Starting Webcam Stream..."
        elif source.startswith('http') or source.startswith('rtsp'):
            source_display = 'IP Camera'
            initial_status = "Starting IP/URL Stream..."
        else:
             # This handles uploaded file paths, though usually handled by /upload_video
             source_display = 'File'
             initial_status = "Starting File Analysis..."
        
        # Update the status state manager to reflect the attempt to start
        # The generate_frames() loop will update the status to "Source Error" if the connection fails.
        global_state.update(last_known_status=initial_status)

        return jsonify({
            "status": initial_status, 
            "is_streaming": True, 
            "video_source": source
        }), 200
    
    elif action == 'stop' and global_state.get("is_streaming"):
        global_state.update(is_streaming=False, last_known_status="Stream Stopped")
        return jsonify({"status": "Stream stopped", "is_streaming": False}), 200

    return jsonify({"status": "Invalid action or already in state", "is_streaming": global_state.get("is_streaming")}), 200


@app.route('/analytics_data')
def analytics_data():
    """Returns real-time analytics data and recent events as JSON."""
    
    current_state = global_state.state # Access the dict directly for a snapshot
    # Handling potential None for last_event_time
    last_event_time_str = current_state["last_event_time"].isoformat() if current_state["last_event_time"] else None
    
    return jsonify({
        'people_count': current_state["last_people_count"],
        'weapon_count': current_state["last_weapon_count"],
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
        global_state.update(last_known_status="Starting File Analysis...")

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

    # Calculate paths based on the project root structure
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alert_source = os.path.join(project_root, 'alert.wav')
    alert_dest = os.path.join(static_path, 'alert.wav')
    
    # Placeholder path configuration
    placeholder_source = os.path.join(project_root, 'placeholder.jpeg')
    placeholder_dest = os.path.join(static_path, 'placeholder.jpeg')
    
    try:
        if os.path.exists(alert_source) and not os.path.exists(alert_dest):
            shutil.copy2(alert_source, alert_dest)
        
        # Ensure placeholder image is available in the static folder
        # NOTE: If placeholder.jpeg doesn't exist in the project root, this will fail silently.
        if os.path.exists(placeholder_source) and not os.path.exists(placeholder_dest):
            shutil.copy2(placeholder_source, placeholder_dest)
            
    except Exception as e:
        print(f"Warning: Could not copy static files (alert.wav or placeholder.jpeg). Ensure they exist. Error: {e}")

    app.run(host='0.0.0.0', port=5000, threaded=True)
