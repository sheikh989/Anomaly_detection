import sys
import os
import time
import cv2
import torch
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from utils.load_model import load_models
from utils.utils import build_transforms
from network.TorchUtils import get_torch_device
from yolo_detection import analyze_video_with_yolo
from video_sumrrizer import summarize_video
import threading
from dotenv import load_dotenv

#---- API Key ----
load_dotenv()
api = os.getenv("OPENAI_API_KEY")


# ---- Config ----
FEATURE_EXTRACTOR_PATH = r"pretrained\\c3d.pickle"
AD_MODEL_PATH = r"exps\\c3d\\models\\epoch_80000.pt"
YOLO_MODEL_PATH = r"yolo_my_model.pt"
DEVICE = get_torch_device()
ANOMALY_THRESHOLD = 0.5
SAVE_DIR = "outputs/anomaly_frames"
os.makedirs(SAVE_DIR, exist_ok=True)
TRANSFORMS = build_transforms(mode="c3d")


# ---- Memory logging ----
# import psutil
# def log_memory_usage(tag="Runtime"):
#     cpu_mem = psutil.virtual_memory().used / (1024 ** 3)
#     gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
#     print(f"[{tag}] CPU: {cpu_mem:.2f} GB | GPU: {gpu_mem:.2f} GB")


# ---- Matplotlib Canvas for anomaly graph ----
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


# ---- Video Processing Thread ----
class VideoProcessor(QThread):
    anomaly_signal = pyqtSignal(float)
    yolo_signal = pyqtSignal(str)
    yolo_frame_signal = pyqtSignal(np.ndarray)
    recording_signal = pyqtSignal(bool)
    summary_signal = pyqtSignal(str)

    def __init__(self, video_path: str, feature_extractor: torch.nn.Module, anomaly_detector: torch.nn.Module):
        super().__init__()
        self.video_path = video_path
        self.feature_extractor = feature_extractor.eval()
        self.anomaly_detector = anomaly_detector.eval()
        self.device = DEVICE
        self.running = True
        self.last_save_time = 0
        self.scores = []

    def stop(self):
        self.running = False

    def smooth_score(self, new_score, window=5):
        self.scores.append(new_score)
        if len(self.scores) > window:
            self.scores.pop(0)
        return float(np.mean(self.scores))

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_buffer = []
        frame_count = 0
        warmup_frames = 0

        if not cap.isOpened():
            print(f"[ERROR] Cannot open video {self.video_path}")
            return

        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_buffer.append(frame.copy())

            # Log memory periodically
            # if frame_count % 100 == 0:
            #     log_memory_usage(f"Frame {frame_count}")

            if len(frame_buffer) == 16:
                frames_resized = [cv2.resize(f, (112, 112)) for f in frame_buffer]
                clip_np = np.array(frames_resized, dtype=np.uint8)
                clip_torch = torch.from_numpy(clip_np)
                clip_torch = TRANSFORMS(clip_torch)
                clip_torch = clip_torch.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    features = self.feature_extractor(clip_torch).detach()
                    score_tensor = self.anomaly_detector(features).detach()
                    score = float(score_tensor.view(-1)[0].item())

                score = self.smooth_score(score)
                score = float(np.clip(score, 0, 1))
                self.anomaly_signal.emit(score)

                # Log after anomaly model
                # if frame_count % 100 == 0:
                #     log_memory_usage("[Anomaly Model Inference]")

                # Detect anomaly
                if score > ANOMALY_THRESHOLD and (time.time() - self.last_save_time) >= 60:
                    self.last_save_time = time.time()
                    self.recording_signal.emit(True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clip_dir = os.path.join(SAVE_DIR, f"anomaly_{timestamp}")
                    os.makedirs(clip_dir, exist_ok=True)

                    # Save first frame (or 3rd) for YOLO
                    yolo_frame_path = os.path.join(clip_dir, "yolo_frame.jpg")
                    third_frame_idx = min(2, len(frame_buffer)-1)
                    cv2.imwrite(yolo_frame_path, frame_buffer[third_frame_idx])

                    # log_memory_usage("[Before YOLO Inference]")

                    # YOLO detection on 3rd frame
                    try:
                        yolo_result = analyze_video_with_yolo(yolo_frame_path, model_path=YOLO_MODEL_PATH, return_class=True)
                        label_text = f"Anomaly Detected → YOLO Class: {yolo_result}"
                    except Exception as e:
                        label_text = f"YOLO Error: {e}"

                    self.yolo_signal.emit(label_text)
                    self.yolo_frame_signal.emit(frame_buffer[third_frame_idx])
                    # log_memory_usage("[After YOLO Inference]")

                    # Record 30-second video for summarizer
                    def save_clip_and_summarize(video_path, start_frame_buffer, clip_dir):
                        try:
                            cap2 = cv2.VideoCapture(video_path)
                            fps = cap2.get(cv2.CAP_PROP_FPS) or 25.0
                            width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            out_path = os.path.join(clip_dir, "anomaly_clip.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                            # Write existing frames first
                            for f in start_frame_buffer:
                                out.write(f)

                            # Then write frames until 30 seconds
                            frames_to_save = int(fps * 30)
                            saved = len(start_frame_buffer)
                            while saved < frames_to_save:
                                ret2, f2 = cap2.read()
                                if not ret2:
                                    break
                                out.write(f2)
                                saved += 1

                            out.release()
                            cap2.release()
                            print(f"[INFO]  Saved 30-sec anomaly clip → {out_path}")

                            # Summarize video
                            summary = summarize_video(out_path,api)
                            self.summary_signal.emit(summary)
                        except Exception as e:
                            print(f"[ERROR] Failed saving/summarizing clip: {e}")

                    threading.Thread(target=save_clip_and_summarize, args=(self.video_path, frame_buffer.copy(), clip_dir), daemon=True).start()

                frame_buffer.clear()
        cap.release()


# ---- Main Pipeline Window ----
class PipelineWindow(QWidget):
    def __init__(self, anomaly_detector, feature_extractor):
        super().__init__()
        self.anomaly_detector = anomaly_detector
        self.feature_extractor = feature_extractor
        self._y_pred = []
        self.video_path = None
        self.processor = None
        self.recording_label = QLabel("")
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Real-Time Anomaly Detection with YOLO")
        self.setGeometry(200, 100, 1000, 700)
        layout = QGridLayout(self)

        # Video Player
        self.videoPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        video_widget = QVideoWidget()
        layout.addWidget(video_widget, 0, 0, 5, 6)
        self.videoPlayer.setVideoOutput(video_widget)

        # Graph
        self.graphWidget = MplCanvas(self, width=6, height=2, dpi=100)
        layout.addWidget(self.graphWidget, 5, 0, 1, 4)

        # YOLO Result
        self.yolo_label = QLabel("YOLO: waiting...")
        self.yolo_label.setStyleSheet("color: lime; font-weight: bold; font-size: 16px;")
        layout.addWidget(self.yolo_label, 6, 0, 1, 4)

        # Preview Frame (YOLO)
        self.image_label = QLabel("YOLO Frame Preview")
        self.image_label.setStyleSheet("border: 2px solid lime;")
        self.image_label.setFixedSize(240, 160)
        layout.addWidget(self.image_label, 6, 4, 1, 2)

        # 30-sec video summary
        self.summary_label = QLabel("Summary: waiting...")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label, 7, 0, 1, 6)

        # Recording Text
        self.recording_label.setText("")
        self.recording_label.setStyleSheet("color: red; font-size: 20px; font-weight: bold;")
        layout.addWidget(self.recording_label, 8, 0, 1, 6)

        # Buttons
        openBtn = QPushButton("Open Video")
        playBtn = QPushButton("Play")
        resetBtn = QPushButton("Reset")

        openBtn.clicked.connect(self.open_video)
        playBtn.clicked.connect(self.play_video)
        resetBtn.clicked.connect(self.reset_pipeline)

        layout.addWidget(openBtn, 9, 0)
        layout.addWidget(playBtn, 9, 1)
        layout.addWidget(resetBtn, 9, 2)

    def open_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video")
        if file:
            self.video_path = file
            self.videoPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file)))
            self.reset_pipeline(clear_video=False)

    def play_video(self):
        if not self.video_path:
            return
        self.videoPlayer.play()
        if self.processor:
            self.processor.stop()
            self.processor.wait()
        self.processor = VideoProcessor(self.video_path, self.feature_extractor, self.anomaly_detector)
        self.processor.anomaly_signal.connect(self.update_graph)
        self.processor.yolo_signal.connect(self.update_yolo_label)
        self.processor.yolo_frame_signal.connect(self.update_yolo_preview)
        self.processor.recording_signal.connect(self.toggle_recording_text)
        self.processor.summary_signal.connect(self.update_summary)
        self.processor.start()

    def update_graph(self, score: float):
        self._y_pred.append(score)
        ax = self.graphWidget.axes
        ax.clear()
        ax.plot(self._y_pred, "r-", linewidth=2)
        ax.axhline(ANOMALY_THRESHOLD, color="yellow", linestyle="--")
        ax.set_ylim(-0.1, 1.1)
        ax.set_title("Live Anomaly Score")
        self.graphWidget.draw()

    def update_yolo_label(self, text: str):
        self.yolo_label.setText(text)

    def update_yolo_preview(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def update_summary(self, text: str):
        self.summary_label.setText(f"Summary: {text}")

    def toggle_recording_text(self, recording: bool):
        self.recording_label.setText(" Anomaly Video Recording Started..." if recording else "")

    def reset_pipeline(self, clear_video=True):
        if self.processor:
            self.processor.stop()
            self.processor.wait()
            self.processor = None
        if clear_video:
            self.videoPlayer.stop()
            self.videoPlayer.setMedia(QMediaContent())
        self._y_pred.clear()
        self.yolo_label.setText("YOLO: waiting...")
        self.image_label.clear()
        self.graphWidget.axes.clear()
        self.graphWidget.draw()
        self.recording_label.setText("")
        self.summary_label.setText("Summary: waiting...")
        print("[INFO] Pipeline reset complete.")

    def closeEvent(self, event):
        if self.processor:
            self.processor.stop()
            self.processor.wait(2000)
        event.accept()


if __name__ == "__main__":
    print("[INFO] Loading models...")
    anomaly_detector, feature_extractor = load_models(
        FEATURE_EXTRACTOR_PATH, AD_MODEL_PATH, features_method="c3d", device=DEVICE
    )

    app = QApplication(sys.argv)
    window = PipelineWindow(anomaly_detector, feature_extractor)
    window.show()
    sys.exit(app.exec_())
