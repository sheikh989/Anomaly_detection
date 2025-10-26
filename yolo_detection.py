import argparse
import os
import sys
import cv2
from ultralytics import YOLO
import numpy as np

# --- Configuration ---
# Set the path for the YOLO model weights
YOLO_MODEL_PATH = r"S:\ano_dec_pro\AnomalyDetectionCVPR2018-Pytorch\yolo_my_model.pt"

# --- Main Detection Function ---
def analyze_video_with_yolo(video_path: str, model_path: str = YOLO_MODEL_PATH, conf_threshold: float = 0.5,return_class=False):
    """
    Analyzes a video using a pre-trained YOLO model for object detection 
    and prints the predicted anomaly class for frames with detections.

    :param video_path: Path to the input video file.
    :param model_path: Path to the trained YOLO model weights.
    :param conf_threshold: Minimum confidence score required for a detection.
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found at: {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] YOLO model not found at: {model_path}")
        return

    try:
        # Load the YOLO model (Assumes it's trained for your 14 anomaly classes)
        model = YOLO(model_path, task='detect')
        labels = model.names
        print(f"[INFO] YOLO Model loaded successfully with {len(labels)} classes.")
    except Exception as e:
        print(f"[FATAL] Failed to load YOLO model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream for: {video_path}")
        return

    frame_num = 0
    detections_found = 0

    print("-" * 50)
    print(f"Starting analysis of: {os.path.basename(video_path)}")
    print("-" * 50)


    detected_class = "None"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % 10 == 0:
            results = model(frame, conf=conf_threshold, verbose=False)
            if results and len(results[0].boxes) > 0:
                detections = results[0].boxes
                best_detection = detections[detections.conf.argmax()]
                class_idx = int(best_detection.cls.item())
                class_name = labels.get(class_idx, "Unknown Class")
                detected_class = class_name
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

    if return_class:
        return detected_class





    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     # Analyze every 10th frame to speed up testing and reduce redundancy
    #     if frame_num % 10 == 0:
            
    #         # Run inference (set verbose=False to keep the output clean)
    #         results = model(frame, conf=conf_threshold, verbose=False)
            
    #         # Process results
    #         if results and len(results[0].boxes) > 0:
    #             detections = results[0].boxes
                
    #             # We'll take the highest confidence detection in the frame
    #             best_detection = detections[detections.conf.argmax()]
                
    #             class_idx = int(best_detection.cls.item())
    #             class_name = labels.get(class_idx, "Unknown Class")
    #             confidence = best_detection.conf.item()
                
    #             print(f"Frame {frame_num:05d}: DETECTED -> {class_name} ({confidence:.2f})")
    #             detections_found += 1

    #     frame_num += 1

    # cap.release()
    # cv2.destroyAllWindows()
    
    print("-" * 50)
    print(f"Analysis complete. Total detections reported: {detections_found}")
    print("-" * 50)


# if __name__ == '__main__':
#     # --- Example Usage (Run this command in your environment) ---
#     # Change the video path to a file you want to test!
#     example_video = r"S:\ano_dec_pro\dataset\Abuse\Abuse002_x264.mp4"
    
#     # NOTE: In a shell/CLI environment (like VS Code or PowerShell), you typically
#     # run this by executing the script and passing arguments:
#     # python yolo_detection.py --video_path "S:\ano_dec_pro\dataset\Abuse\Abuse002_x264.mp4" 
    
#     # We call the function directly for simple execution in Python interpreter
#     print("NOTE: Running the detection script. Please change 'example_video' to your actual file path.")
#     analyze_video_with_yolo(example_video)
