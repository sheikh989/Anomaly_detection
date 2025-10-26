import sys
import os
import time
import cv2
import torch
import numpy as np
import threading
from collections import deque
from datetime import datetime
from typing import List, Optional
from queue import Queue, Empty

from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QPushButton, QLabel, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from utils.load_model import load_models
from utils.utils import build_transforms
from network.TorchUtils import get_torch_device
from yolo_detection import analyze_video_with_yolo
from video_sumrrizer import summarize_video
from dotenv import load_dotenv

#---- API Key ----
load_dotenv()
api = os.getenv("OPENAI_API_KEY")
# ---- CONFIG ----
FEATURE_EXTRACTOR_PATH = r"pretrained\\c3d.pickle"
AD_MODEL_PATH = r"exps\\c3d\\models\\epoch_80000.pt"
YOLO_MODEL_PATH = r"yolo_my_model.pt"

DEVICE = get_torch_device()
ANOMALY_THRESHOLD = 0.5
SAVE_DIR = "outputs/anomaly_clips"
os.makedirs(SAVE_DIR, exist_ok=True)

CLIP_FRAMES = 16                 # 16-frame clip for C3D
ANOMALY_RECORD_SECONDS = 30      # record 30 seconds after anomaly
COOLDOWN_SECONDS = 150           # min seconds between anomaly saves
TRANSFORMS = build_transforms(mode="c3d")
# ---------------------------------------

# small helper: matplotlib canvas
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


# Capture thread: reads frames via OpenCV, emits preview frames and appends into shared deque.
class CaptureThread(QThread):
    preview_frame = pyqtSignal(np.ndarray)  # BGR frame
    stopped = pyqtSignal()

    def __init__(self, video_path: str, shared_deque: deque, recorder_queue_ref: dict):
        super().__init__()
        self.video_path = video_path
        self.shared_deque = shared_deque      # deque for processing (shared)
        self.recorder_queue_ref = recorder_queue_ref  # dict {'queue': Optional[Queue]} - mutable ref
        self._running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"[ERROR] CaptureThread: cannot open {self.video_path}")
            self.stopped.emit()
            return

        # Optionally set preview smaller to reduce CPU load
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # emit preview (main GUI will convert to QPixmap)
            try:
                self.preview_frame.emit(frame.copy())
            except Exception:
                pass

            # push to processing deque (thread-safe if using only append/pop with GIL)
            if self.shared_deque is not None:
                self.shared_deque.append(frame.copy())

            # if registry queue exists (recorder), put frame for recorder
            q = self.recorder_queue_ref.get('queue', None)
            if q is not None:
                # don't block: if queue full, drop frame (we want real-time)
                try:
                    q.put_nowait(frame.copy())
                except Exception:
                    pass

            # small sleep to avoid pegging CPU; fine-tune for your target fps
            time.sleep(1/30)  # ~30 FPS


        self.cap.release()
        self.stopped.emit()

    def stop(self):
        self._running = False
        self.wait(1000)


# Recorder thread (writes frames from a queue into an mp4 file). Uses a blocking queue with timeout.
class RecorderThread(threading.Thread):
    def __init__(self, q: Queue, out_path: str, fps: float, duration_seconds: int):
        super().__init__(daemon=True)
        self.q = q
        self.out_path = out_path
        self.fps = fps
        self.duration_seconds = duration_seconds
        self.stop_event = threading.Event()

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # get one frame size by peeking (wait awhile)
        frame = None
        try:
            frame = self.q.get(timeout=2.0)
        except Empty:
            print("[Recorder] no initial frame - abort")
            return

        h, w, _ = frame.shape
        out = cv2.VideoWriter(self.out_path, fourcc, max(1.0, self.fps), (w, h))

        # write first frame
        out.write(frame)

        frames_to_save = int(self.fps * self.duration_seconds)
        saved = 1

        start = time.time()
        while saved < frames_to_save and not self.stop_event.set():
            try:
                frame = self.q.get(timeout=2.0)
                out.write(frame)
                saved += 1
            except Empty:
                # if no frame for some time, break
                print("[Recorder] queue empty while recording")
                break

        out.release()
        print(f"[Recorder] finished -> {self.out_path}")

    def stop(self):
        self._stop.set()


# Processor thread: consumes the shared deque periodically, runs feature_extractor+anomaly detector.
class ProcessorThread(QThread):
    anomaly_score = pyqtSignal(float)
    yolo_label_signal = pyqtSignal(str)
    yolo_frame_signal = pyqtSignal(np.ndarray)
    recording_signal = pyqtSignal(bool)
    summary_signal = pyqtSignal(str)

    def __init__(self, shared_deque: deque, capture_thread: CaptureThread,
                 feature_extractor: torch.nn.Module, anomaly_detector: torch.nn.Module):
        super().__init__()
        self.shared_deque = shared_deque
        self.feature_extractor = feature_extractor.eval()
        self.anomaly_detector = anomaly_detector.eval()
        self.device = DEVICE
        self.capture_thread = capture_thread
        self._running = True
        self._last_saved_time = 0.0
        self.score_window: List[float] = []

        # recorder state
        self.recorder_queue_ref = capture_thread.recorder_queue_ref

    def stop(self):
        self._running = False
        self.wait(1000)

    def smooth(self, val: float, window=5):
        self.score_window.append(val)
        if len(self.score_window) > window:
            self.score_window.pop(0)
        return float(sum(self.score_window) / len(self.score_window))

    def run(self):
        # We'll poll the deque at a small interval and process the latest CLIP_FRAMES
        while self._running:
            try:
                if len(self.shared_deque) >= CLIP_FRAMES:
                    # copy last CLIP_FRAMES frames
                    # IMPORTANT: do this under small critical section
                    clip = []
                    # copy from deque tail
                    tail = list(self.shared_deque)[-CLIP_FRAMES:]
                    for f in tail:
                        clip.append(f)
                    # prepare for model
                    frames_resized = [cv2.resize(f, (112, 112)) for f in clip]
                    clip_np = np.array(frames_resized, dtype=np.uint8)
                    clip_torch = torch.from_numpy(clip_np)
                    clip_torch = TRANSFORMS(clip_torch)
                    clip_torch = clip_torch.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        feat = self.feature_extractor(clip_torch).detach()
                        pred = self.anomaly_detector(feat).detach()

                    if pred.numel() == 1:
                        score = float(pred.item())
                    else:
                        score = float(pred.view(-1)[0].item())

                    score = self.smooth(score)
                    score = float(np.clip(score, 0.0, 1.0))
                    self.anomaly_score.emit(score)

                    now = time.time()
                    if score > ANOMALY_THRESHOLD and (now - self._last_saved_time) >= COOLDOWN_SECONDS:
                        # START anomaly actions
                        self._last_saved_time = now
                        self.recording_signal.emit(True)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        clip_dir = os.path.join(SAVE_DIR, f"anomaly_{timestamp}")
                        os.makedirs(clip_dir, exist_ok=True)

                        # 3rd frame index in clip (index 2) or last available
                        y3_idx = min(2, len(clip) - 1)
                        yolo_frame = clip[y3_idx].copy()
                        yolo_img_path = os.path.join(clip_dir, "yolo_frame.jpg")
                        cv2.imwrite(yolo_img_path, yolo_frame)

                        # emit frame preview to UI
                        self.yolo_frame_signal.emit(yolo_frame.copy())

                        # Run YOLO on saved image (this runs in processor thread; capture thread still running)
                        try:
                            yolo_label = analyze_video_with_yolo(yolo_img_path, model_path=YOLO_MODEL_PATH, return_class=True)
                            yolo_text = f"YOLO: {yolo_label}"
                        except Exception as e:
                            yolo_text = f"YOLO error: {e}"
                            print("[ProcessorThread] YOLO failed:", e)

                        self.yolo_label_signal.emit(yolo_text)

                        # Start recorder: create a new recorder queue and give it to capture thread
                        recorder_q = Queue(maxsize=1024)  # for recorded frames
                        self.recorder_queue_ref['queue'] = recorder_q

                        # capture fps from capture Thread's cv2 capture if possible
                        fps = self.capture_thread.cap.get(cv2.CAP_PROP_FPS) if self.capture_thread.cap else 25.0
                        out_path = os.path.join(clip_dir, "anomaly_clip.mp4")
                        recorder = RecorderThread(recorder_q, out_path, fps or 25.0, ANOMALY_RECORD_SECONDS)
                        recorder.start()

                        # After recorder finishes, it will have saved video; but we do not block here.
                        # We'll run a watcher thread to wait for recorder then run summarizer and emit results.
                        def watcher_and_summarize(rec_thread: RecorderThread, out_video_path: str, clip_dir_local: str):
                            # wait for recorder to finish (join)
                            rec_thread.join(timeout=(ANOMALY_RECORD_SECONDS + 10))
                            # release recorder queue from capture thread
                            self.recorder_queue_ref['queue'] = None
                            # Now summarize video (this takes time); do in separate thread to avoid blocking processor thread loop
                            try:
                                summary = summarize_video(out_video_path,api)
                            except Exception as e:
                                summary = f"Summarizer error: {e}"
                                print("[Watcher] summarizer failed:", e)
                            # emit summary
                            self.summary_signal.emit(summary)
                            # signal recording stopped
                            self.recording_signal.emit(False)

                        # start watcher
                        threading.Thread(target=watcher_and_summarize, args=(recorder, out_path, clip_dir), daemon=True).start()

                    # end of anomaly check

                # sleep small amount to avoid busy looping
                time.sleep(0.02)
            except Exception as e:
                print("[ProcessorThread] error:", e)
                time.sleep(0.1)

        # cleanup
        self.recorder_queue_ref['queue'] = None


# MAIN GUI window (keeps preview, graph, yolo preview, summary, record text, open/play/reset)
class PipelineWindow(QWidget):
    def __init__(self, anomaly_detector, feature_extractor):
        super().__init__()
        self.anomaly_detector = anomaly_detector
        self.feature_extractor = feature_extractor

        # shared deque used between capture & processor
        self.shared_deque = deque(maxlen=CLIP_FRAMES * 10)  # holds recent frames
        self.recorder_queue_ref = {'queue': None}           # mutable ref accessible by threads

        self.capture_thread: Optional[CaptureThread] = None
        self.processor_thread: Optional[ProcessorThread] = None

        # UI state
        self._y_pred: List[float] = []

        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Real-Time Anomaly Detection (OpenCV Preview)")
        self.setGeometry(200, 100, 1000, 700)

        layout = QGridLayout(self)

        # big preview
        self.preview_label = QLabel("Video Preview")
        self.preview_label.setFixedSize(640, 360)
        self.preview_label.setStyleSheet("background:black;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label, 0, 0, 5, 4)

        # graph
        self.graph_widget = MplCanvas(self, width=6, height=2, dpi=100)
        layout.addWidget(self.graph_widget, 5, 0, 1, 4)

        # YOLO label + preview
        self.yolo_label = QLabel("YOLO: waiting...")
        self.yolo_label.setStyleSheet("color: blue; font-weight: bold;font-size:18px;")
        layout.addWidget(self.yolo_label, 0, 4, 1, 2)

        self.yolo_preview = QLabel("YOLO Frame")
        self.yolo_preview.setFixedSize(240, 160)
        self.yolo_preview.setStyleSheet("border:1px solid blue;")
        self.yolo_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.yolo_preview, 1, 4, 2, 2)

        # summary
        self.summary_label = QLabel("Summary: waiting...")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label, 3, 4, 2, 2)

        # recording text label
        self.recording_label = QLabel("")
        self.recording_label.setStyleSheet("color:red; font-weight:bold; font-size:16px;")
        layout.addWidget(self.recording_label, 5, 4, 1, 2)

        # buttons
        open_btn = QPushButton("Open Video")
        play_btn = QPushButton("Play / Start")
        reset_btn = QPushButton("Reset")
        open_btn.clicked.connect(self.open_file)
        play_btn.clicked.connect(self.toggle_play)
        reset_btn.clicked.connect(self.reset_pipeline)
        layout.addWidget(open_btn, 6, 0)
        layout.addWidget(play_btn, 6, 1)
        layout.addWidget(reset_btn, 6, 2)

        # local flags
        self.playing = False

    def open_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Video")
        if not f:
            return
        self.video_path = f
        self.reset_pipeline(clear_video=False)
        self.yolo_label.setText("YOLO: waiting...")
        self.summary_label.setText("Summary: waiting...")

    def toggle_play(self):
        if not hasattr(self, 'video_path') or self.video_path is None:
            return
        if not self.playing:
            self.start_all()
        else:
            self.stop_all()

    def start_all(self):
        # stop previous if any
        self.stop_all()

        # capture thread
        self.capture_thread = CaptureThread(self.video_path, self.shared_deque, self.recorder_queue_ref)
        self.capture_thread.preview_frame.connect(self.on_preview_frame)
        self.capture_thread.stopped.connect(self.on_capture_stopped)
        self.capture_thread.start()

        # processor thread
        self.processor_thread = ProcessorThread(self.shared_deque, self.capture_thread,
                                                feature_extractor=self.feature_extractor,
                                                anomaly_detector=self.anomaly_detector)
        self.processor_thread.anomaly_score.connect(self.on_anomaly_score)
        self.processor_thread.yolo_label_signal.connect(self.on_yolo_label)
        self.processor_thread.yolo_frame_signal.connect(self.on_yolo_frame)
        self.processor_thread.recording_signal.connect(self.on_recording)
        self.processor_thread.summary_signal.connect(self.on_summary)
        self.processor_thread.start()

        self.playing = True

    def stop_all(self):
        # stop processor first, then capture
        if self.processor_thread:
            try:
                self.processor_thread.stop()
            except Exception:
                pass
            self.processor_thread = None

        if self.capture_thread:
            try:
                self.capture_thread.stop()
            except Exception:
                pass
            self.capture_thread = None

        # also free recorder queue if present
        if self.recorder_queue_ref.get('queue', None) is not None:
            try:
                self.recorder_queue_ref['queue'] = None
            except Exception:
                pass

        self.playing = False

    # UI slots
    def on_preview_frame(self, frame_bgr: np.ndarray):
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio))
        except Exception as e:
            print("[WARN] preview update failed:", e)

    def on_anomaly_score(self, score: float):
        # update graph
        self._y_pred.append(score)
        ax = self.graph_widget.axes
        ax.clear()
        ax.plot(self._y_pred, "r-", linewidth=2)
        ax.axhline(ANOMALY_THRESHOLD, color="yellow", linestyle="--")
        ax.set_ylim(-0.1, 1.1)
        ax.set_title("Live Anomaly Score")
        self.graph_widget.draw()

    def on_yolo_label(self, text: str):
        self.yolo_label.setText(text)

    def on_yolo_frame(self, frame_bgr: np.ndarray):
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.yolo_preview.setPixmap(pix.scaled(self.yolo_preview.size(), Qt.KeepAspectRatio))
        except Exception as e:
            print("[WARN] yolo preview failed:", e)

    def on_recording(self, recording: bool):
        if recording:
            self.recording_label.setText("📹 Recording anomaly (30s)...")
        else:
            self.recording_label.setText("")

    def on_summary(self, text: str):
        self.summary_label.setText("Summary: " + text)

    def on_capture_stopped(self):
        print("[PipelineWindow] capture stopped")

    def reset_pipeline(self, clear_video=True):
        # stop threads
        self.stop_all()
        if clear_video:
            if hasattr(self, 'video_path'):
                del self.video_path
        self._y_pred.clear()
        self.preview_label.clear()
        self.yolo_preview.clear()
        self.yolo_label.setText("YOLO: waiting...")
        self.summary_label.setText("Summary: waiting...")
        self.recording_label.setText("")
        self.graph_widget.axes.clear()
        self.graph_widget.draw()
        self.shared_deque.clear()
        self.recorder_queue_ref['queue'] = None
        print("[INFO] Pipeline reset complete.")

    def closeEvent(self, event):
        self.reset_pipeline()
        event.accept()


if __name__ == "__main__":
    print("[INFO] Loading models...")
    anomaly_detector, feature_extractor = load_models(
        FEATURE_EXTRACTOR_PATH, AD_MODEL_PATH, features_method="c3d", device=DEVICE
    )
    print("[INFO] Models loaded.")

    app = QApplication(sys.argv)
    win = PipelineWindow(anomaly_detector, feature_extractor)
    win.show()
    sys.exit(app.exec_())
