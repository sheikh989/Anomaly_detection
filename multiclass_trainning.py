"""Video Anomaly Classification (13 Classes)"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils import data
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.video_utils import VideoClips

# Category mapping
# ---------------------------
folder_to_category = {
    "Normal": 0,
    "Abuse": 1,
    "Arrest": 2,
    "Arson": 3,
    "Assault": 4,
    "Burglary": 5,
    "Explosion": 6,
    "Fighting": 7,
    "Robbery": 8,
    "Shooting": 9,
    "Shoplifting": 10,
    "Stealing": 11,
    "Vandalism": 12,
}

anomaly_categories = {v: k for k, v in folder_to_category.items()}

# Video Dataset Loader (OpenCV)
# ---------------------------
class VideoIter(data.Dataset):
    """Custom Video Loader using OpenCV."""

    def __init__(self, clip_length, frame_stride,
                 dataset_path=None, video_transform=None, return_label=False):
        super().__init__()
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.video_transform = video_transform
        self.dataset_path = dataset_path
        self.return_label = return_label
        self.video_list = self._get_video_list(dataset_path)

    def _get_video_list(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        vid_list = []
        for path, _, files in os.walk(dataset_path):
            for name in files:
                if name.lower().endswith((".mp4", ".avi", ".mov")):
                    vid_list.append(os.path.join(path, name))
        logging.info(f"Found {len(vid_list)} videos in {dataset_path}")
        return vid_list

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_path = self.video_list[index]
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % self.frame_stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            count += 1
            if len(frames) >= self.clip_length:
                break
        cap.release()

        # pad if too short
        while len(frames) < self.clip_length:
            frames.append(frames[-1])

        video = np.array(frames)  # (T,H,W,C)
        if self.video_transform is not None:
            video = torch.stack([self.video_transform(f) for f in video])  # (T,C,H,W)

        dir, file = video_path.split(os.sep)[-2:]
        file = file.split(".")[0]

        if self.return_label:
            label = folder_to_category.get(dir, 0)
            return video, label, 0, dir, file
        return video, 0, dir, file

# Model Definition
# ---------------------------
class AnomalyClassifier(nn.Module):
    """13-class anomaly classification model."""

    def __init__(self, input_dim=512, num_classes=13):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)

# Feature Extractor (ResNet18)
# ---------------------------
def build_feature_extractor(device):
    resnet = models.resnet18(pretrained=True)
    modules = list(resnet.children())[:-1]  # remove final fc
    backbone = nn.Sequential(*modules).to(device)
    backbone.eval()
    return backbone


def extract_features(backbone, frames, device):
    """frames: (T,C,H,W)"""
    with torch.no_grad():
        feats = backbone(frames.to(device))  # (T,512,1,1)
        feats = feats.view(feats.size(0), -1)  # (T,512)
        feats = feats.mean(dim=0)  # (512,)
    return feats

# Training
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Video Anomaly Classification (13 classes, ResNet features)")
    parser.add_argument("--dataset_path", type=str, default="/content/dataset",
                        help="path to dataset root folder")
    parser.add_argument("--exps_dir", type=str, default="/content/exps",
                        help="directory to save models and logs")
    parser.add_argument("--save_every", type=int, default=1,
                        help="epochs interval for saving checkpoints")
    parser.add_argument("--lr_base", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    return parser.parse_args(args=[])


def train(args):
    os.makedirs(args.exps_dir, exist_ok=True)
    models_dir = os.path.join(args.exps_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VideoIter(
        clip_length=16,
        frame_stride=8,
        dataset_path=args.dataset_path,
        video_transform=transform,
        return_label=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)

    # feature extractor + classifier
    backbone = build_feature_extractor(device)
    model = AnomalyClassifier(input_dim=512, num_classes=13).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i, batch in enumerate(train_loader):
            videos, labels, _, _, _ = batch  # videos: (B,T,C,H,W)
            batch_feats = []
            for vid in videos:  # loop videos
                feat = extract_features(backbone, vid, device)  # (512,)
                batch_feats.append(feat)
            batch_feats = torch.stack(batch_feats).to(device)  # (B,512)

            labels = labels.to(device)

            outputs = model(batch_feats)          # (B,13)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] finished. Avg Loss = {avg_loss:.4f}, Acc = {acc:.2f}%")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(models_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    args = get_args()
    train(args)
 



