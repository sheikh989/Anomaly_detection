import os
import cv2
import torch
import numpy as np
import time
from datetime import datetime
import threading
import base64
from werkzeug.utils import secure_filename

# --- Load .env file ---
from dotenv import load_dotenv
load_dotenv()
# --- END ---

from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO

# Important: Adjust imports if your structure changed
from utils.load_model import load_models
from utils.utils import build_transforms
from network.TorchUtils import get_torch_device
from yolo_detection import analyze_video_with_yolo
from video_sumrrizer import summarize_video # Your summarizer

# ---- App Setup ----
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key!'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app, async_mode='eventlet')

# ---- Global Config & Model Loading ----
print("[INFO] Loading models...")
DEVICE = get_torch_device()
FEATURE_EXTRACTOR_PATH = "pretrained/c3d.pickle"
AD_MODEL_PATH = "exps/c3d/models/epoch_80000.pt"
YOLO_MODEL_PATH = "yolo_my_model.pt"
SAVE_DIR = "outputs/anomaly_frames"
ANOMALY_THRESHOLD = 0.4
COOLDOWN_SECS = 60.0 # <--- This enforces the 60 second anomaly cooldown
os.makedirs(SAVE_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY: print("WARNING: OPENAI_API_KEY not set!")

anomaly_detector, feature_extractor = load_models( FEATURE_EXTRACTOR_PATH, AD_MODEL_PATH, features_method="c3d", device=DEVICE)
feature_extractor.eval(); anomaly_detector.eval()
TRANSFORMS = build_transforms(mode="c3d")
print("[INFO] Models loaded successfully.")

VIDEO_PATHS = { # Ensure these paths are correct relative to app.py
    "Abuse": "demo_videos/Abuse.mp4", "Arrest": "demo_videos/Arrest.mp4",
    "Arson": "demo_videos/Arson.mp4", "Assault": "demo_videos/Assault.mp4",
    "Burglary": "demo_videos/Burglary.mp4", "Explosion": "demo_videos/Explosion.mp4",
    "Fighting": "demo_videos/Fighting.mp4", "RoadAccidents": "demo_videos/RoadAccidents.mp4",
    "Robbery": "demo_videos/Robbery.mp4", "Shooting": "demo_videos/Shooting.mp4",
    "Shoplifting": "demo_videos/Shoplifting.mp4", "Stealing": "demo_videos/Stealing.mp4",
    "Vandalism": "demo_videos/Vandalism.mp4", "Normal": "demo_videos/Normal.mp4"
}

# --- Thread control ---
thread = None; thread_lock = threading.Lock(); stop_event = threading.Event()

def smooth_score(scores, new_score, window=5):
    scores.append(new_score); scores = scores[-window:]; return float(np.mean(scores))

# --- MODIFIED: Renamed and removed internal wait ---
def _save_clip_and_start_summarizer(video_path, clip_dir, initial_frames, fps, width, height):
    """
    Saves a 30s clip, then immediately starts the summarizer task.
    """
    out_path = os.path.join(clip_dir, "anomaly_clip.mp4")
    save_success = False
    try:
        socketio.emit('update_status', {'status': 'Saving 30s clip...'})
        cap2 = cv2.VideoCapture(video_path)
        if not cap2.isOpened(): raise Exception(f"Cannot open video {video_path}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for f_idx, f in enumerate(initial_frames):
            out.write(f)
            if f_idx % 5 == 0: socketio.sleep(0.001)

        frames_to_save = int(fps * 30)
        remaining = frames_to_save - len(initial_frames)

        for i in range(remaining):
            ret, frame = cap2.read()
            if not ret: break
            out.write(frame)
            if i % 5 == 0: socketio.sleep(0.001)

        out.release(); cap2.release()
        print(f"[INFO] Saved 30-sec clip -> {out_path}")
        save_success = True

    except Exception as e:
        print(f"[ERROR] _save_clip failed: {e}")
        socketio.emit('update_summary', {'summary': f"Error saving clip: {e}"})
        socketio.emit('recording_signal', {'recording': False})

    finally:
        # --- If saving succeeded, start summarizer immediately ---
        if save_success:
            try:
                print("Starting background task for summarization...")
                socketio.emit('update_status', {'status': 'Clip saved. Starting summarizer...'})
                # Start summarizer in a *new* background task
                socketio.start_background_task(_run_summarizer, out_path)
            except Exception as e:
                 print(f"[ERROR] Failed to start summarizer task: {e}")
                 socketio.emit('update_summary', {'summary': f"Error starting summarizer: {e}"})
                 socketio.emit('recording_signal', {'recording': False})
        # --- End ---

# --- Background Task 2 - Just Run Summarizer (Unchanged) ---
def _run_summarizer(saved_clip_path):
    """
    Runs the summarizer on the already saved clip. Emits results and recording=False.
    """
    summary_text = None
    try:
        print(f" [INFO] Summarizing clip: {saved_clip_path}")
        socketio.emit('update_status', {'status': 'Summarizing clip...'})
        socketio.sleep(0.01) # Yield before blocking call

        api_key_from_env = os.getenv("OPENAI_API_KEY")
        if not api_key_from_env: raise ValueError("OPENAI_API_KEY missing.")

        summary_text = summarize_video(video_path=saved_clip_path, api=api_key_from_env)
        print("\n VIDEO SUMMARY (snippet):\n", summary_text[:100] + "...", "\n")

    except Exception as e:
        summary_text = f"Summarizer Error: {e}"
        print(f"[ERROR] _run_summarizer failed: {e}")
    finally:
        if summary_text is None: summary_text = "Summarizer Error: Unknown failure."
        socketio.emit('update_summary', {'summary': summary_text})
        socketio.emit('recording_signal', {'recording': False}) # Signal recording finished
        print("ℹ Summarization processing finished.")


# --- Main video processing loop (Unchanged from previous fix) ---
def video_processing_task(video_path):
    global thread
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise Exception("Cannot open video file")

        frame_buffer, last_save_time, recent_scores = [], 0.0, []
        FRAME_SKIP = 6 # Keep increased frame skipping
        frame_count = 0
        fps, width, height = (cap.get(cv2.CAP_PROP_FPS) or 25.0,
                              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        while cap.isOpened() and not stop_event.is_set():
            socketio.sleep(0.002) # Keep yielding

            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            if frame_count % (FRAME_SKIP + 1) != 0: continue # Apply skipping

            frame_buffer.append(frame.copy())

            if len(frame_buffer) == 16:
                socketio.sleep(0.001) # Yield before inference

                # --- Anomaly Detection ---
                frames_resized = [cv2.resize(f, (112, 112)) for f in frame_buffer]
                clip_np = np.array(frames_resized, dtype=np.uint8)
                clip_torch = torch.from_numpy(clip_np)
                clip_torch = TRANSFORMS(clip_torch).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    features = feature_extractor(clip_torch).detach()
                    score = float(anomaly_detector(features).detach().item())
                score = smooth_score(recent_scores, score)
                score = float(np.clip(score, 0, 1))
                socketio.emit('update_graph', {'score': score})
                # --- End Anomaly Detection ---

                # --- Anomaly Actions ---
                now = time.time()
                # --- This check enforces the 60 second cooldown ---
                if score > ANOMALY_THRESHOLD and (now - last_save_time) >= COOLDOWN_SECS:
                    print(f" Anomaly! Score: {score:.2f}")
                    last_save_time = now # Reset cooldown timer

                    socketio.emit('recording_signal', {'recording': True})
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clip_dir = os.path.join(SAVE_DIR, f"anomaly_{timestamp}")
                    os.makedirs(clip_dir, exist_ok=True)

                    # Quick YOLO
                    yolo_frame_path = os.path.join(clip_dir, "yolo_frame.jpg")
                    preview_frame_img = frame_buffer[-1].copy()
                    cv2.imwrite(yolo_frame_path, preview_frame_img)
                    try:
                        yolo_result = analyze_video_with_yolo(yolo_frame_path, model_path=YOLO_MODEL_PATH, return_class=True)
                        yolo_text = f"🚨 Anomaly → YOLO: {yolo_result}"
                    except Exception as e: yolo_text = f"YOLO Error: {e}"
                    socketio.emit('update_yolo_text', {'text': yolo_text})
                    _, buffer = cv2.imencode('.jpg', preview_frame_img)
                    b64_str = base64.b64encode(buffer).decode('utf-8')
                    socketio.emit('update_yolo_image', {'image_data': b64_str})

                    socketio.emit('update_summary', {'summary': 'loading'})
                    socketio.sleep(0.005) # Yield before starting save task

                    # Start the saving task (which will later schedule summarizer)
                    print(" Starting background task for clip saving...")
                    socketio.start_background_task(
                        _save_clip_and_start_summarizer, # Use the modified function
                        video_path, clip_dir, frame_buffer.copy(),
                        fps, width, height
                    )
                # --- End anomaly actions ---
                frame_buffer.clear()

        cap.release()
        if not stop_event.is_set(): socketio.emit('processing_finished', {'message': 'Video finished.'})
        print(" Video processing task ended.")

    except Exception as e:
         print(f"[ERROR] Unhandled exception in video_processing_task: {e}")
         socketio.emit('processing_error', {'error': f'Runtime error: {e}'})
    finally:
        with thread_lock:
            if thread is not None:
                 print("🧹 Cleaning up video processing task.")
                 thread = None; stop_event.clear()

# --- (Routes: /, /upload, /video_stream remain the same) ---
@app.route('/')
def index(): return render_template('index.html', anomaly_names=VIDEO_PATHS.keys())

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files: return jsonify({'error': 'No video file'}), 400
    file = request.files['video']; name = file.filename
    if name == '': return jsonify({'error': 'No video selected'}), 400
    filename = secure_filename(name); ts = datetime.now().strftime('%Y%m%d%H%M%S')
    unique_name = f"{ts}_{filename}"; path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    try: file.save(path); print(f" Uploaded: {path}"); return jsonify({'success': True, 'filename': unique_name})
    except Exception as e: print(f" Upload failed: {e}"); return jsonify({'error': f'{e}'}), 500

@app.route('/video_stream/<source>/<path:filename>')
def video_stream(source, filename):
    path, safe_name = None, secure_filename(filename)
    if source == 'demo': path = VIDEO_PATHS.get(filename)
    elif source == 'upload': path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    else: return "Invalid source", 404
    if not path or not os.path.exists(path): return "Video not found", 404
    def generate():
        try:
            with open(path, "rb") as f:
                while chunk := f.read(1024*1024): yield chunk; socketio.sleep(0.0001)
        except Exception as e: print(f" Streaming error: {e}")
    print(f" Streaming: {path}"); return Response(generate(), mimetype="video/mp4")

# --- (SocketIO handlers: start_processing, reset_system remain the same) ---
@socketio.on('start_processing')
def handle_start_processing(data):
    global thread
    with thread_lock:
        if thread is None:
            stop_event.clear(); source, name = data.get('source'), data.get('filename')
            path, safe_name = None, secure_filename(name) if name else None
            if not safe_name: return socketio.emit('processing_error', {'error': 'Invalid filename.'})
            if source == 'demo': path = VIDEO_PATHS.get(name)
            elif source == 'upload': path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            if path and os.path.exists(path):
                print(f"Starting processing: '{safe_name}' ({source})")
                thread = socketio.start_background_task(video_processing_task, path)
            else: socketio.emit('processing_error', {'error': 'Video file not found.'})
        else: socketio.emit('update_status', {'status': 'Processing already running.'})

@socketio.on('reset_system')
def handle_reset():
    global thread
    with thread_lock:
        if thread: print(" Reset requested. Stopping..."); stop_event.set()
        else: print("Reset requested, none running.")
    socketio.emit('system_reset_confirm'); print(" Reset confirmed.")

if __name__ == '__main__':
    print("Starting Flask server...")
    socketio.run(app, debug=True)