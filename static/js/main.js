document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    // Get all UI elements
    const videoPlayer = document.getElementById('videoPlayer');
    const yoloTextLabel = document.getElementById('yoloTextLabel');
    const yoloImageFrame = document.getElementById('yoloImageFrame');
    const statusLabel = document.getElementById('statusLabel');
    const resetButton = document.getElementById('resetButton');
    const anomalySelector = document.getElementById('anomalySelector');
    const videoUploadInput = document.getElementById('videoUpload');
    const uploadButton = document.getElementById('uploadButton');
    
    // Get the new summary elements
    const summaryTextLabel = document.getElementById('summaryTextLabel');
    const summaryLoaderContainer = document.getElementById('summaryLoaderContainer');

    let chart;

    function initializeChart() {
        const ctx = document.getElementById('anomalyChart').getContext('2d');
        if (chart) { chart.destroy(); }
        chart = new Chart(ctx, {
            type: 'line', data: { labels: [], datasets: [{ label: 'Anomaly Score', data: [], borderColor: 'rgba(255, 99, 132, 1)', backgroundColor: 'rgba(255, 99, 132, 0.2)', borderWidth: 2, tension: 0.4, pointRadius: 0 }] }, options: { scales: { y: { beginAtZero: true, max: 1.0, ticks: { color: '#e0e0e0' }}, x: { ticks: { color: '#e0e0e0' }}}, plugins: { legend: { labels: { color: '#e0e0e0' }}}}
        });
    }

    // This function resets the whole UI
    function resetUI() {
        videoPlayer.pause();
        videoPlayer.removeAttribute('src');
        videoPlayer.load();
        initializeChart();
        yoloTextLabel.textContent = 'Waiting for anomaly...';
        yoloImageFrame.src = '';
        statusLabel.textContent = 'System reset. Select a video to begin.';
        videoUploadInput.value = '';
        anomalySelector.selectedIndex = 0;
        
        // Reset the summary box
        summaryLoaderContainer.style.display = 'none'; // Hide loader
        summaryTextLabel.style.display = 'block'; // Show text
        summaryTextLabel.textContent = 'Summary will appear here after analysis...';
    }

    // --- WebSocket Event Listeners ---
    socket.on('connect', () => { statusLabel.textContent = 'Connected. Please select a video to start processing.'; });
    socket.on('update_graph', (data) => {
        const { score } = data;
        if (!chart) return;
        const newLabel = chart.data.labels.length + 1;
        chart.data.labels.push(newLabel);
        chart.data.datasets[0].data.push(score);
        if (chart.data.labels.length > 100) { chart.data.labels.shift(); chart.data.datasets[0].data.shift(); }
        chart.update();
    });
    socket.on('update_yolo_text', (data) => { yoloTextLabel.textContent = data.text; });
    socket.on('update_yolo_image', (data) => { yoloImageFrame.src = `data:image/jpeg;base64,${data.image_data}`; });
    socket.on('update_status', (data) => { statusLabel.textContent = data.status; });
    socket.on('processing_error', (data) => { statusLabel.textContent = `Error: ${data.error}`; });
    socket.on('processing_finished', (data) => { statusLabel.textContent = data.message; });
    socket.on('system_reset_confirm', () => { resetUI(); });

    // NEW: Listener for the recording light
    socket.on('recording_signal', (data) => {
        if (data.recording) {
            statusLabel.textContent = "Anomaly detected! Recording 30s clip...";
        } 
        // The 'false' signal is handled by the summary logic
    });

    // --- NEW: Logic for the summary box ---
    socket.on('update_summary', (data) => {
        const summary = data.summary;
        
        if (summary === 'loading') {
            // Show the loader
            summaryTextLabel.style.display = 'none';
            summaryLoaderContainer.style.display = 'flex';
        } else {
            // Hide the loader and show the final summary
            summaryLoaderContainer.style.display = 'none';
            summaryTextLabel.style.display = 'block';
            summaryTextLabel.textContent = summary; // Set the final text
            statusLabel.textContent = 'Analysis complete.'; // Update status
        }
    });

    // --- User Interaction (No changes) ---
    anomalySelector.addEventListener('change', (event) => {
        const anomalyName = event.target.value;
        if (!anomalyName) return; 
        resetUI();
        statusLabel.textContent = `Requesting to process ${anomalyName}...`;
        videoPlayer.src = `/video_stream/demo/${anomalyName}`;
        videoPlayer.play();
        socket.emit('start_processing', { 'source': 'demo', 'filename': anomalyName });
    });

    resetButton.addEventListener('click', () => { socket.emit('reset_system'); });

    uploadButton.addEventListener('click', () => {
        const file = videoUploadInput.files[0];
        if (!file) {
            alert('Please select a video file first!');
            return;
        }
        resetUI();
        statusLabel.textContent = 'Uploading video...';
        const formData = new FormData();
        formData.append('video', file);

        fetch('/upload', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const uploadedFilename = data.filename;
                statusLabel.textContent = `Upload successful. Starting analysis...`;
                videoPlayer.src = `/video_stream/upload/${uploadedFilename}`;
                videoPlayer.play();
                socket.emit('start_processing', { 'source': 'upload', 'filename': uploadedFilename });
            } else {
                statusLabel.textContent = `Error: ${data.error}`;
                alert(`Upload failed: ${data.error}`);
            }
        })
        .catch(error => {
            statusLabel.textContent = 'An error occurred during upload.';
            console.error('Upload error:', error);
        });
    });

    // Initialize the chart and UI on page load
    initializeChart();
    resetUI(); // Call reset to ensure correct initial state
});