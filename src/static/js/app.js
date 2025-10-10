document.addEventListener('DOMContentLoaded', function() {
    // --- CONTROL ELEMENTS ---
    const startWebcamBtn = document.getElementById('start-webcam-btn');
    const stopStreamBtn = document.getElementById('stop-stream-btn');
    const streamStatusMessage = document.getElementById('stream-status-message');
    
    // IP elements were removed from HTML and JS.
    
    const videoStreamImg = document.getElementById('video-stream-img');
    const streamOfflineOverlay = document.getElementById('stream-offline-overlay');
    
    const uploadForm = document.getElementById('upload-form');
    const videoFileInput = document.getElementById('video-file-input');
    const fileNameDisplay = document.getElementById('file-name-display');
    const startBatchBtn = document.getElementById('start-batch-analysis-btn');
    const uploadMessage = document.getElementById('upload-message');
    const uploadError = document.getElementById('upload-error');

    // --- METRIC & LOG ELEMENTS ---
    const videoPreview = document.getElementById('video-preview');
    const videoPreviewContainer = document.getElementById('video-preview-container');

    const peopleCount = document.getElementById('people-count');
    const weaponCountEl = document.getElementById('weapon-count'); 
    const currentStatus = document.getElementById('current-status');
    const currentStatusBox = document.getElementById('current-status-box');
    const eventLogContainer = document.getElementById('event-log-container');
    
    const alertAudio = document.getElementById('alert-audio');

    // API endpoints (derived from Flask context in index.html, or using defaults)
    const PLACEHOLDER_URL = videoStreamImg.dataset.placeholder || "/static/placeholder.jpeg";
    const VIDEO_FEED_URL = "/video_feed";
    const ANALYTICS_URL = "/analytics_data";
    const TOGGLE_URL = "/toggle_stream";
    const UPLOAD_URL = "/upload_video";
    const LOG_FILES_URL_BASE = "/log_files/TYPE/FILENAME";
    
    let lastEventTimestamp = null;
    let isUploading = false;
    let pollingInterval = 500;
    let pollingTimer = null;


    // --- Utility Functions ---

    function updateControls(isStreaming, source) {
        // Source can only be 'webcam' or a file path
        const isFileAnalysis = source !== 'webcam' && !source.startsWith('http') && !source.startsWith('rtsp');

        // Control the main start buttons
        startWebcamBtn.disabled = isStreaming;
        stopStreamBtn.disabled = !isStreaming;
        
        // Control the File Analysis button
        startBatchBtn.disabled = isStreaming || isUploading || videoFileInput.files.length === 0;

        // Visual stream management
        if (isStreaming) {
            streamOfflineOverlay.classList.add('hidden');
            // Ensure the video stream source is set to the Flask endpoint with a cache buster
            if (videoStreamImg.src.indexOf(VIDEO_FEED_URL) === -1 || videoStreamImg.src.indexOf(PLACEHOLDER_URL) !== -1) {
                videoStreamImg.src = `${VIDEO_FEED_URL}?${new Date().getTime()}`; 
            }
        } else {
            streamOfflineOverlay.classList.remove('hidden');
            videoStreamImg.src = PLACEHOLDER_URL;

            // Re-enable start button if a stream isn't active
            startWebcamBtn.disabled = false;
        }
        
        // Disable file upload while any stream is running
        videoFileInput.disabled = isStreaming;

        // Update stream status message based on source
        if (isStreaming && isFileAnalysis) {
             streamStatusMessage.textContent = `Analyzing File: ${source}`;
        } else if (isStreaming && source === 'webcam') {
             streamStatusMessage.textContent = 'Streaming from Webcam';
        }
    }

    function updateStatusUI(status, isStreaming) {
        currentStatus.textContent = status;
        currentStatusBox.className = 'p-3 rounded-lg transition-colors duration-500';
        
        switch (status) {
            case "ANOMALY CONFIRMED":
                currentStatusBox.classList.add('status-anomaly');
                streamStatusMessage.textContent = 'ANOMALY DETECTED! Processing alert...';
                break;
            case "Potential Anomaly":
                currentStatusBox.classList.add('status-potential');
                streamStatusMessage.textContent = 'Potential anomaly detected. Monitoring closely.';
                break;
            case "Normal":
                currentStatusBox.classList.add('status-normal');
                streamStatusMessage.textContent = isStreaming ? 'Stream Active: Monitoring...' : 'Normal. Stream Stopped.';
                break;
            case "Source Error": 
                currentStatusBox.classList.add('status-error');
                streamStatusMessage.textContent = 'ERROR: Could not connect to camera source. Check device.';
                break;
            case "Starting Webcam Stream...":
                currentStatusBox.classList.add('status-potential');
                streamStatusMessage.textContent = 'Starting local webcam...';
                break;
            case "File Analysis Complete":
            case "Analysis Finished":
                currentStatusBox.classList.add('status-stopped');
                streamStatusMessage.textContent = 'Analysis Complete. Ready for next file.';
                break;
            default: // Awaiting Input, Stream Stopped
                currentStatusBox.classList.add('status-stopped');
                streamStatusMessage.textContent = status;
                break;
        }
    }

    function playAlert(eventTimestamp) {
        if (eventTimestamp && eventTimestamp !== lastEventTimestamp) {
            alertAudio.currentTime = 0; 
            alertAudio.play().catch(e => console.warn("Audio blocked by browser policy:", e));
            lastEventTimestamp = eventTimestamp;
        }
    }
    
    function renderEventLog(events) {
        eventLogContainer.innerHTML = '';
        
        if (events.length === 0) {
            eventLogContainer.innerHTML = '<p id="no-events-message" class="text-gray-500">No recent anomalies detected.</p>';
            return;
        }

        const noEventsEl = document.getElementById('no-events-message');
        if (noEventsEl) noEventsEl.remove();


        events.slice().reverse().forEach(event => {
            const frameUrl = LOG_FILES_URL_BASE.replace('TYPE', 'frame').replace('FILENAME', event.frame_file);
            const clipDownloadUrl = LOG_FILES_URL_BASE.replace('TYPE', 'clip').replace('FILENAME', event.clip_file); // Retaining download link
            
            const weaponCount = event.weapons_detected || 0;
            const threatColor = weaponCount > 0 ? 'text-red-800' : 'text-orange-600';
            const threatIcon = weaponCount > 0 ? 'HIGH THREAT' : 'BEHAVIORAL'; // Changed to 'BEHAVIORAL' since 'Weapon Detected' is redundant if weapon count > 0

            const eventCard = document.createElement('div');
            eventCard.className = 'bg-white p-4 rounded-xl shadow-md border border-red-300 transition-shadow hover:shadow-lg';
            
            eventCard.innerHTML = `
                <div class="flex items-center space-x-3 mb-2">
                    <svg class="w-6 h-6 ${threatColor}" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.398 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                    <p class="font-bold text-lg ${threatColor}">ANOMALY: ${threatIcon}</p>
                </div>
                <p class="text-sm text-gray-700 mb-3">Timestamp: ${event.timestamp}</p>
                
                <div class="grid grid-cols-2 gap-4">
                    <div class="space-y-1">
                        <div class="flex justify-between text-sm">
                            <span class="text-gray-600">Confidence:</span>
                            <span class="font-semibold text-red-700">${event.confidence}</span>
                        </div>
                        <div class="flex justify-between text-sm">
                            <span class="text-gray-600">People:</span>
                            <span class="font-semibold text-blue-700">${event.people_detected}</span>
                        </div>
                        <div class="flex justify-between text-sm">
                            <span class="text-gray-600">Weapons:</span>
                            <span class="font-semibold ${weaponCount > 0 ? 'text-red-700' : 'text-green-700'}">${weaponCount}</span>
                        </div>
                         <a href="${clipDownloadUrl}" download class="inline-block mt-3 text-xs text-blue-500 hover:underline">Download Video Clip</a>
                    </div>

                    <div class="border rounded-lg overflow-hidden">
                        <img src="${frameUrl}" alt="Trigger Frame" class="w-full h-auto object-cover" onclick="window.open('${frameUrl}', '_blank')">
                    </div>
                </div>
            `;
            eventLogContainer.appendChild(eventCard);
        });
    }


    // --- API and Main Loop ---

    function fetchAnalyticsData() {
        fetch(ANALYTICS_URL)
            .then(response => {
                // Handle 204 No Content response for stopped streams
                if (response.status === 204) {
                     return { 
                        people_count: 0, 
                        weapon_count: 0, // Ensure default is 0 for consistency
                        status: "Stream Stopped", 
                        is_streaming: false, 
                        video_source: "webcam", 
                        events: [],
                        last_event_time: null 
                    };
                }
                if (!response.ok) throw new Error('Network response was not ok');
                return response.json();
            })
            .then(data => {
                // 1. Update Metrics
                peopleCount.textContent = data.people_count;
                updateStatusUI(data.status, data.is_streaming);

                // FIX: Update the dedicated weapon count metric using the real-time data.weapon_count field
                weaponCountEl.textContent = data.weapon_count || 0; 

                // 2. Control State
                updateControls(data.is_streaming, data.video_source);

                // 3. Audio Alert 
                if (data.status === "ANOMALY CONFIRMED") {
                    playAlert(data.last_event_time);
                }

                // 4. Update Event Log 
                renderEventLog(data.events);
            })
            .catch(error => {
                console.error('Error fetching analytics:', error);
                updateControls(false, 'webcam'); 
                updateStatusUI("API Error", false);
            });
    }

    // --- Core Toggle Stream Function ---
    async function toggleStream(action, source_url = 'webcam') {
        if (action === 'start') {
            const currentState = await fetch(ANALYTICS_URL).then(res => res.json()).catch(() => ({is_streaming: false}));
            if (currentState.is_streaming) {
                console.error("Please stop the current stream first.");
                return;
            }
        }
        
        // Disable buttons immediately
        startWebcamBtn.disabled = true;
        stopStreamBtn.disabled = true;

        // Update status UI immediately to give feedback
        const startingStatus = 'Starting Webcam Stream...';
        updateStatusUI(action === 'start' ? startingStatus : 'Stream Stopped.', false);


        try {
            const response = await fetch(TOGGLE_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // Pass the source URL (always 'webcam' or the file path) to the backend
                body: JSON.stringify({ action: action, source: source_url }) 
            });
        } catch (error) {
            console.error('Failed to toggle stream:', error);
            updateStatusUI("Connection Error", false);
        }
    }

    // --- Event Listeners for Controls ---

    // 1. Listener for starting local webcam (sends 'webcam' index 0)
    startWebcamBtn.addEventListener('click', () => toggleStream('start', 'webcam'));
    
    // 2. Listener for stopping any stream
    stopStreamBtn.addEventListener('click', () => toggleStream('stop'));
    
    // --- Video File Input and Preview ---
    videoFileInput.addEventListener('change', () => {
        uploadMessage.classList.add('hidden');
        uploadError.classList.add('hidden');
        
        if (videoFileInput.files.length > 0) {
            const file = videoFileInput.files[0];
            fileNameDisplay.textContent = file.name;
            
            const fileURL = URL.createObjectURL(file);
            videoPreview.src = fileURL;
            videoPreviewContainer.classList.remove('hidden');

            // Disable webcam start button when a file is selected (for clarity)
            startWebcamBtn.disabled = true;

        } else {
            fileNameDisplay.textContent = 'No file chosen';
            videoPreview.src = '';
            videoPreviewContainer.classList.add('hidden');
            // Re-enable control buttons
            fetchAnalyticsData();
        }
    });

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (videoFileInput.files.length === 0) return;

        isUploading = true;
        startBatchBtn.disabled = true;
        uploadMessage.classList.add('hidden');
        uploadError.classList.add('hidden');
        fileNameDisplay.textContent = 'Uploading...';
        
        const formData = new FormData(uploadForm);
        
        try {
            const response = await fetch(UPLOAD_URL, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                uploadMessage.textContent = data.message;
                uploadMessage.classList.remove('hidden');
                
                // Clear the upload UI fields
                videoFileInput.value = ''; 
                fileNameDisplay.textContent = 'No file chosen';
                videoPreview.src = '';
                videoPreviewContainer.classList.add('hidden');

            } else {
                uploadError.textContent = data.message || "Upload failed due to a server error.";
                uploadError.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Upload fetch failed:', error);
            uploadError.textContent = "Upload failed: Network or server connection lost.";
            uploadError.classList.remove('hidden');
        } finally {
            isUploading = false;
        }
    });

    // --- Initialization ---
    // Hide the messages immediately on load/refresh
    uploadMessage.classList.add('hidden');
    uploadError.classList.add('hidden');
    
    // Initial load state
    fetchAnalyticsData();

    // Start polling 
    if (pollingTimer) clearInterval(pollingTimer);
    pollingTimer = setInterval(fetchAnalyticsData, pollingInterval);
});