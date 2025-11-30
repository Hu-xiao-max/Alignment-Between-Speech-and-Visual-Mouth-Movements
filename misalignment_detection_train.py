"""
Misalignment Detection - Training Script
用法: python misalignment_detection_train.py --max_samples 200 --epochs 20
"""

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from torch.utils.data import Dataset, DataLoader

from dataset import GridDataset
from model import LipNet


class Logger:
    """Log to both file and optionally to console"""
    def __init__(self, log_path: str, console: bool = False):
        self.log_path = log_path
        self.console = console
        self.file = open(log_path, 'w')
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.file.write(line + "\n")
        self.file.flush()
        if self.console:
            print(message)
    
    def close(self):
        self.file.close()


def format_time(seconds: float) -> str:
    """Format seconds into human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DetectorConfig:
    img_width: int = 100
    img_height: int = 50
    max_video_length: int = 75
    sample_rate: int = 16000
    n_mfcc: int = 20
    max_shift_frames: int = 10
    num_negative_samples: int = 1
    default_fps: float = 25.0


def get_video_fps(video_path: str, fallback: float = 25.0) -> float:
    if video_path.endswith(".npy"):
        return fallback
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 1e-3 else fallback


def shift_audio(audio: np.ndarray, shift_frames: int, fps: float, sample_rate: int) -> np.ndarray:
    if shift_frames == 0:
        return audio.copy()
    shift_samples = int(shift_frames / max(fps, 1e-5) * sample_rate)
    if shift_samples == 0:
        return audio.copy()
    result = np.zeros_like(audio)
    if shift_samples > 0:
        if shift_samples < len(audio):
            result[shift_samples:] = audio[:-shift_samples]
    else:
        shift_samples = abs(shift_samples)
        if shift_samples < len(audio):
            result[:-shift_samples] = audio[shift_samples:]
    return result


def compute_audio_stats(audio: np.ndarray, sample_rate: int, n_mfcc: int) -> torch.Tensor:
    if audio.size == 0:
        return torch.zeros(n_mfcc * 2, dtype=torch.float32)
    hop_length = max(1, int(sample_rate / 40))
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length)
    if mfcc.size == 0:
        return torch.zeros(n_mfcc * 2, dtype=torch.float32)
    mfcc_tensor = torch.from_numpy(mfcc.T).float()
    mean = mfcc_tensor.mean(dim=0)
    std = mfcc_tensor.std(dim=0)
    return torch.cat([mean, std], dim=0)


def extract_visual_embeddings(lipnet: LipNet, frames: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x = F.relu(lipnet.conv1(frames))
        x = lipnet.pool1(x)
        x = lipnet.dropout1(x)
        x = F.relu(lipnet.conv2(x))
        x = lipnet.pool2(x)
        x = lipnet.dropout2(x)
        x = F.relu(lipnet.conv3(x))
        x = lipnet.pool3(x)
        x = lipnet.dropout3(x)
        batch_size, channels, time_steps, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, time_steps, -1)
    return x


class FeatureExtractor:
    def __init__(self, grid_dataset: GridDataset, lipnet: LipNet, device: torch.device, cfg: DetectorConfig):
        self.grid = grid_dataset
        self.lipnet = lipnet.to(device)
        self.device = device
        self.cfg = cfg
        self.visual_cache: dict = {}
        self.audio_cache: dict = {}
        self.fps_cache: dict = {}

    def _load_visual_stats(self, video_path: str) -> Tuple[torch.Tensor, float]:
        if video_path in self.visual_cache:
            return self.visual_cache[video_path], self.fps_cache[video_path]
        frames = self.grid.process_video(video_path)
        fps = get_video_fps(video_path, self.cfg.default_fps)
        frame_tensor = frames.unsqueeze(0).to(self.device)
        embeddings = extract_visual_embeddings(self.lipnet, frame_tensor)
        emb = embeddings.squeeze(0).cpu()
        stats = torch.cat([emb.mean(dim=0), emb.std(dim=0)], dim=0)
        self.visual_cache[video_path] = stats
        self.fps_cache[video_path] = fps
        return stats, fps

    def _load_audio(self, video_path: str) -> Tuple[np.ndarray, int]:
        if video_path in self.audio_cache:
            return self.audio_cache[video_path]
        
        # Try librosa first
        try:
            audio, sr = librosa.load(video_path, sr=None)
        except Exception:
            # Fallback to moviepy for .mpg files
            try:
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(video_path)
                if clip.audio is None:
                    clip.close()
                    raise RuntimeError(f"No audio in {video_path}")
                sr = clip.audio.fps
                audio = clip.audio.to_soundarray(fps=sr)
                clip.close()
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
            except Exception as e:
                raise RuntimeError(f"Failed to load audio from {video_path}: {e}")
        
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        audio = audio.astype(np.float32)
        self.audio_cache[video_path] = (audio, sr)
        return audio, sr

    def build_feature(self, video_path: str, shift_frames: int) -> Tuple[torch.Tensor, dict]:
        visual_stats, fps = self._load_visual_stats(video_path)
        audio, sr = self._load_audio(video_path)
        if sr != self.cfg.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.cfg.sample_rate)
            sr = self.cfg.sample_rate
        shifted_audio = shift_audio(audio, shift_frames, fps, sr)
        audio_stats = compute_audio_stats(shifted_audio, sr, self.cfg.n_mfcc)
        feature = torch.cat([visual_stats, audio_stats], dim=0)
        return feature, {"video_path": video_path, "shift_frames": shift_frames, "fps": fps}


class MisalignmentDataset(Dataset):
    def __init__(self, video_paths: List[str], extractor: FeatureExtractor, cfg: DetectorConfig, seed: int = 0):
        self.video_paths = video_paths
        self.extractor = extractor
        self.cfg = cfg
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.video_paths) * (1 + self.cfg.num_negative_samples)

    def __getitem__(self, idx: int):
        base_idx = idx // (1 + self.cfg.num_negative_samples)
        variant_idx = idx % (1 + self.cfg.num_negative_samples)
        video_path = self.video_paths[base_idx]
        if variant_idx == 0:
            shift_frames = 0
            label = 1.0
        else:
            magnitude = self.rng.randint(1, max(1, self.cfg.max_shift_frames))
            direction = self.rng.choice([-1, 1])
            shift_frames = magnitude * direction
            label = 0.0
        feature, _ = self.extractor.build_feature(video_path, shift_frames)
        return feature, torch.tensor(label, dtype=torch.float32)


class MisalignmentDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        probs = torch.sigmoid(logits)
        total_loss += loss.item() * features.size(0)
        all_labels.append(labels.detach().cpu())
        all_probs.append(probs.detach().cpu())
    
    labels_t = torch.cat(all_labels).numpy()
    probs_t = torch.cat(all_probs).numpy()
    preds = (probs_t > 0.5).astype(float)
    acc = accuracy_score(labels_t, preds)
    try:
        auc = roc_auc_score(labels_t, probs_t)
    except ValueError:
        auc = float("nan")
    return {"loss": total_loss / len(dataloader.dataset), "acc": acc, "auc": auc, "labels": labels_t, "probs": probs_t}


def plot_roc(labels: np.ndarray, probs: np.ndarray, out_path: str) -> None:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_lipnet(checkpoint_path: str, vocab_size: int, device: torch.device) -> LipNet:
    lipnet = LipNet(vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        lipnet.load_state_dict(checkpoint["model_state_dict"])
    else:
        lipnet.load_state_dict(checkpoint)
    lipnet.eval()
    for p in lipnet.parameters():
        p.requires_grad = False
    return lipnet


def save_detector(model: MisalignmentDetector, path: str, cfg: DetectorConfig) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "config": {"sample_rate": cfg.sample_rate, "n_mfcc": cfg.n_mfcc, "max_shift_frames": cfg.max_shift_frames}
    }, path)
    print(f"Detector saved to {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train misalignment detector")
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--checkpoint", type=str, default="lipnet_final.pth")
    p.add_argument("--detector_checkpoint", type=str, default="misalignment_detector.pth")
    p.add_argument("--speakers", nargs="*", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_shift_frames", type=int, default=15)
    p.add_argument("--num_negatives", type=int, default=1)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_mfcc", type=int, default=20)
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", help="Also print logs to console")
    p.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs") #--------------------------
    return p.parse_args()


def main():
    args = parse_args()
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = os.path.join(args.log_dir, f"misalignment_{timestamp}")
    os.makedirs(log_folder, exist_ok=True)
    
    # Initialize logger
    log_path = os.path.join(log_folder, "training.log")
    logger = Logger(log_path, console=args.verbose)
    
    # Start timing
    start_time = time.time()
    
    logger.log("=" * 60)
    logger.log("Misalignment Detection Training")
    logger.log("=" * 60)
    logger.log(f"Log folder: {log_folder}")
    logger.log(f"Arguments: {vars(args)}")
    
    set_seed(args.seed)
    device = get_device()
    logger.log(f"Using device: {device}")
    print(f"Using device: {device}")
    print(f"Logs will be saved to: {log_folder}")

    cfg = DetectorConfig(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        max_shift_frames=args.max_shift_frames,
        num_negative_samples=args.num_negatives,
    )

    speakers = args.speakers or sorted([d for d in os.listdir(args.data_path) if d.startswith("s")])
    base_dataset = GridDataset(args.data_path, speakers, img_width=cfg.img_width, img_height=cfg.img_height, max_video_length=cfg.max_video_length)
    
    video_paths = [v for v, _ in base_dataset.samples]
    if args.max_samples:
        random.shuffle(video_paths)
        video_paths = video_paths[:args.max_samples]
    logger.log(f"Using {len(video_paths)} videos from {len(speakers)} speakers")
    print(f"Using {len(video_paths)} videos")

    lipnet = load_lipnet(args.checkpoint, len(base_dataset.vocab), device)
    extractor = FeatureExtractor(base_dataset, lipnet, device, cfg)

    # Split data
    random.shuffle(video_paths)
    n = len(video_paths)
    train_paths = video_paths[:int(n * 0.7)]
    val_paths = video_paths[int(n * 0.7):int(n * 0.85)]
    test_paths = video_paths[int(n * 0.85):]
    
    logger.log(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    train_ds = MisalignmentDataset(train_paths, extractor, cfg, seed=args.seed)
    val_ds = MisalignmentDataset(val_paths, extractor, cfg, seed=args.seed + 1)
    test_ds = MisalignmentDataset(test_paths, extractor, cfg, seed=args.seed + 2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    feature_dim = lipnet.conv_output_dim * 2 + cfg.n_mfcc * 2
    model = MisalignmentDetector(feature_dim, args.hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.log("")
    logger.log("Training started...")

    best_state, best_auc = None, -1.0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_m = run_epoch(model, train_loader, criterion, device, optimizer)
        val_m = run_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        log_msg = (f"Epoch {epoch:02d}/{args.epochs} | "
                   f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.3f} | "
                   f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.3f} val_auc={val_m['auc']:.3f} | "
                   f"time={epoch_time:.1f}s")
        logger.log(log_msg)
        
        if not np.isnan(val_m["auc"]) and val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            best_state = model.state_dict()
            logger.log(f"  -> New best model (val_auc={best_auc:.3f})")
        
        # Save checkpoint every N epochs------------------------------------------------------
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(log_folder, f"checkpoint_epoch_{epoch}.pth")
            save_detector(model, checkpoint_path, cfg)
            logger.log(f"  -> Checkpoint saved: {checkpoint_path}")
        # Save checkpoint every N epochs------------------------------------------------------

    if best_state:
        model.load_state_dict(best_state)

    logger.log("")
    logger.log("Evaluating on test set...")
    test_m = run_epoch(model, test_loader, criterion, device)
    logger.log(f"Test -> loss: {test_m['loss']:.4f}, acc: {test_m['acc']:.3f}, auc: {test_m['auc']:.3f}")

    # Save model to log folder
    detector_path = os.path.join(log_folder, os.path.basename(args.detector_checkpoint))
    save_detector(model, detector_path, cfg)
    
    # Also save to the specified path
    save_detector(model, args.detector_checkpoint, cfg)

    # Save ROC curve
    roc_path = os.path.join(log_folder, "roc.png")
    plot_roc(test_m["labels"], test_m["probs"], roc_path)
    logger.log(f"ROC saved to {roc_path}")
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.log("")
    logger.log("=" * 60)
    logger.log(f"Training completed!")
    logger.log(f"Total time: {format_time(total_time)}")
    logger.log(f"Best val AUC: {best_auc:.3f}")
    logger.log(f"Test AUC: {test_m['auc']:.3f}")
    logger.log(f"Model saved to: {args.detector_checkpoint}")
    logger.log(f"Logs saved to: {log_folder}")
    logger.log("=" * 60)
    
    logger.close()
    
    # Print summary to console
    print("")
    print("=" * 60)
    print(f"Training completed!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best val AUC: {best_auc:.3f}")
    print(f"Test AUC: {test_m['auc']:.3f}")
    print(f"Model saved to: {args.detector_checkpoint}")
    print(f"Logs saved to: {log_folder}")
    print("=" * 60)


if __name__ == "__main__":
    main()