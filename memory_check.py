import psutil
import torch
from utils.load_model import load_models
from network.TorchUtils import get_torch_device
from yolo_detection import analyze_video_with_yolo  # if you have YOLO loading inside this
from ultralytics import YOLO  # if you use Ultralytics YOLOv11

FEATURE_EXTRACTOR_PATH = r"S:\ano_dec_pro\AnomalyDetectionCVPR2018-Pytorch\pretrained\c3d.pickle"
AD_MODEL_PATH = r"S:\ano_dec_pro\AnomalyDetectionCVPR2018-Pytorch\exps\c3d\models\epoch_80000.pt"
YOLO_MODEL_PATH = r"S:\ano_dec_pro\AnomalyDetectionCVPR2018-Pytorch\yolo_my_model.pt"

DEVICE = get_torch_device()


def get_memory_usage():
    """Return (CPU_used_GB, GPU_used_GB)."""
    cpu_mem = psutil.virtual_memory().used / (1024 ** 3)
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem


def print_diff(before, after, name):
    cpu_diff = after[0] - before[0]
    gpu_diff = after[1] - before[1]
    print(f"[{name}] CPU: +{cpu_diff:.2f} GB | GPU: +{gpu_diff:.2f} GB")


print("[INFO] Measuring memory usage step-by-step...\n")

# --- 1️⃣ Base memory before anything ---
base_mem = get_memory_usage()
print(f"[BASELINE] CPU: {base_mem[0]:.2f} GB | GPU: {base_mem[1]:.2f} GB\n")

# --- 2️⃣ Load anomaly detector (C3D + AD model) ---
before_ano = get_memory_usage()
anomaly_detector, feature_extractor = load_models(
    FEATURE_EXTRACTOR_PATH, AD_MODEL_PATH, features_method="c3d", device=DEVICE
)
after_ano = get_memory_usage()
print_diff(before_ano, after_ano, "Anomaly Model")

# --- 3️⃣ Load YOLO model ---
before_yolo = get_memory_usage()
yolo_model = YOLO(YOLO_MODEL_PATH)  # adjust if you use custom loader
after_yolo = get_memory_usage()
print_diff(before_yolo, after_yolo, "YOLO Model")

# --- 4️⃣ Final total memory ---
final_mem = get_memory_usage()
print(f"\n[TOTAL USED] CPU: {final_mem[0] - base_mem[0]:.2f} GB | GPU: {final_mem[1] - base_mem[1]:.2f} GB\n")
