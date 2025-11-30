"""
Misalignment Detection - Demo Generation Script
用法: python misalignment_detection_demo.py --save_demo_dir demos --demo_shift_frames 10
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from moviepy.editor import VideoFileClip, ImageSequenceClip
    from moviepy.audio.AudioClip import AudioArrayClip, AudioClip
    import moviepy.video.fx.all as vfx
    MOVIEPY_AVAILABLE = True
except Exception:
    VideoFileClip = ImageSequenceClip = AudioArrayClip = AudioClip = vfx = None
    MOVIEPY_AVAILABLE = False

from dataset import GridDataset
from model import LipNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def build_demo_audio_track(audio: np.ndarray, shift_frames: int, fps: float, sample_rate: int, duration: float) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(int(duration * sample_rate), dtype=np.float32)
    shifted = shift_audio(audio, shift_frames, fps, sample_rate)
    expected = max(1, int(duration * sample_rate))
    if shifted.shape[0] < expected:
        shifted = np.pad(shifted, (0, expected - shifted.shape[0]))
    else:
        shifted = shifted[:expected]
    return shifted.astype(np.float32)


def compute_audio_stats(audio: np.ndarray, sample_rate: int, n_mfcc: int) -> torch.Tensor:
    if audio.size == 0:
        return torch.zeros(n_mfcc * 2, dtype=torch.float32)
    hop_length = max(1, int(sample_rate / 40))
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length)
    if mfcc.size == 0:
        return torch.zeros(n_mfcc * 2, dtype=torch.float32)
    mfcc_tensor = torch.from_numpy(mfcc.T).float()
    return torch.cat([mfcc_tensor.mean(dim=0), mfcc_tensor.std(dim=0)], dim=0)


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
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b, t, -1)
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
        emb = extract_visual_embeddings(self.lipnet, frames.unsqueeze(0).to(self.device)).squeeze(0).cpu()
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
        self.audio_cache[video_path] = (audio.astype(np.float32), sr)
        return self.audio_cache[video_path]

    def build_feature(self, video_path: str, shift_frames: int) -> Tuple[torch.Tensor, dict]:
        visual_stats, fps = self._load_visual_stats(video_path)
        audio, sr = self._load_audio(video_path)
        if sr != self.cfg.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.cfg.sample_rate)
            sr = self.cfg.sample_rate
        shifted = shift_audio(audio, shift_frames, fps, sr)
        audio_stats = compute_audio_stats(shifted, sr, self.cfg.n_mfcc)
        return torch.cat([visual_stats, audio_stats], dim=0), {"shift_frames": shift_frames, "fps": fps}


class MisalignmentDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


def load_lipnet(path: str, vocab_size: int, device: torch.device) -> LipNet:
    lipnet = LipNet(vocab_size=vocab_size)
    ckpt = torch.load(path, map_location=device)
    lipnet.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    lipnet.eval()
    for p in lipnet.parameters():
        p.requires_grad = False
    return lipnet


def load_detector(path: str, device: torch.device) -> Tuple[MisalignmentDetector, dict]:
    ckpt = torch.load(path, map_location=device)
    model = MisalignmentDetector(ckpt["input_dim"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt.get("config", {})


def annotate_frame_rgb(frame: np.ndarray, text: str) -> np.ndarray:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, text, (10, bgr.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_shifted_audio_clip(clip, shift_frames: int, fps: float, sample_rate: int):
    if clip.audio is None or sample_rate <= 0:
        return None
    orig_sr = getattr(clip.audio, "fps", None) or sample_rate
    arr = clip.audio.to_soundarray(fps=orig_sr)
    if arr.ndim == 1:
        arr = arr[:, None]
    shifted = []
    for ch in range(arr.shape[1]):
        shifted.append(build_demo_audio_track(arr[:, ch], shift_frames, fps, orig_sr, clip.duration))
    return AudioArrayClip(np.stack(shifted, axis=1).astype(np.float32), fps=orig_sr)


def save_demo_with_moviepy(clip, out_path: str, text: str, audio_clip) -> None:
    annotated = clip.fl_image(lambda f: annotate_frame_rgb(f, text))
    if audio_clip is not None:
        annotated = annotated.set_audio(audio_clip)
    else:
        annotated = annotated.without_audio()
    annotated.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    annotated.close()


def export_demo(args, extractor: FeatureExtractor, model: MisalignmentDetector, device: torch.device, video_path: str):
    os.makedirs(args.save_demo_dir, exist_ok=True)

    aligned_feat, _ = extractor.build_feature(video_path, 0)
    misaligned_feat, meta = extractor.build_feature(video_path, args.demo_shift_frames)

    model.eval()
    with torch.no_grad():
        aligned_score = torch.sigmoid(model(aligned_feat.unsqueeze(0).to(device))).item()
        misaligned_score = torch.sigmoid(model(misaligned_feat.unsqueeze(0).to(device))).item()

    print(f"Video: {video_path}")
    print(f"  Aligned score: {aligned_score:.4f}")
    print(f"  Misaligned score (shift={args.demo_shift_frames}): {misaligned_score:.4f}")

    if not MOVIEPY_AVAILABLE:
        print("WARNING: MoviePy not available, cannot generate video demos")
        return

    clip = VideoFileClip(video_path)
    if args.demo_scale != 1.0 and vfx:
        clip = clip.fx(vfx.resize, args.demo_scale)
    fps = clip.fps or 25.0

    sr = args.demo_audio_sample_rate or getattr(clip.audio, "fps", None) or 16000
    sr = int(sr)

    aligned_audio = clip.audio if args.demo_include_audio else None
    misaligned_audio = build_shifted_audio_clip(clip, args.demo_shift_frames, fps, sr) if args.demo_include_audio else None

    save_demo_with_moviepy(clip, os.path.join(args.save_demo_dir, "aligned_demo.mp4"), f"aligned | score: {aligned_score:.2f}", aligned_audio)
    save_demo_with_moviepy(clip, os.path.join(args.save_demo_dir, "misaligned_demo.mp4"), f"shift={args.demo_shift_frames} | score: {misaligned_score:.2f}", misaligned_audio)

    clip.close()
    if misaligned_audio:
        misaligned_audio.close()

    print(f"Saved demos to {args.save_demo_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate misalignment demo videos")
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--checkpoint", type=str, default="lipnet_final.pth")
    p.add_argument("--detector_checkpoint", type=str, default="misalignment_detector.pth")
    p.add_argument("--speakers", nargs="*", default=None)
    p.add_argument("--demo_video", type=str, default=None, help="Specific video path (optional)")
    p.add_argument("--save_demo_dir", type=str, default="demos")
    p.add_argument("--demo_shift_frames", type=int, default=10)
    p.add_argument("--min_shift", type=int, default=5, help="Minimum shift frames")  # 新增
    p.add_argument("--max_shift", type=int, default=20, help="Maximum shift frames")  # 新增
    p.add_argument("--demo_use_raw_video", action="store_true")
    p.add_argument("--demo_include_audio", action="store_true")
    p.add_argument("--demo_scale", type=float, default=2.0)
    p.add_argument("--demo_audio_sample_rate", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Load detector
    model, saved_cfg = load_detector(args.detector_checkpoint, device)
    print(f"Loaded detector from {args.detector_checkpoint}")

    cfg = DetectorConfig(
        sample_rate=saved_cfg.get("sample_rate", 16000),
        n_mfcc=saved_cfg.get("n_mfcc", 20),
        max_shift_frames=saved_cfg.get("max_shift_frames", 10),
    )

    speakers = args.speakers or sorted([d for d in os.listdir(args.data_path) if d.startswith("s")])
    base_dataset = GridDataset(args.data_path, speakers, img_width=cfg.img_width, img_height=cfg.img_height, max_video_length=cfg.max_video_length)

    lipnet = load_lipnet(args.checkpoint, len(base_dataset.vocab), device)
    extractor = FeatureExtractor(base_dataset, lipnet, device, cfg)

    if args.demo_video:
        # 单个视频：随机生成shift
        random_shift = random.randint(args.min_shift, args.max_shift)
        args.demo_shift_frames = random_shift
        print(f"Using random shift: {random_shift}")
        export_demo(args, extractor, model, device, args.demo_video)
    else:
        # 每个speaker随机选一个视频
        videos_by_speaker = {}
        for video_path, _ in base_dataset.samples:
            speaker = os.path.basename(os.path.dirname(video_path))
            if speaker not in videos_by_speaker:
                videos_by_speaker[speaker] = []
            videos_by_speaker[speaker].append(video_path)
        
        print(f"Found {len(videos_by_speaker)} speakers")
        
        for speaker, videos in videos_by_speaker.items():
            video_path = random.choice(videos)
            
            # 每个speaker随机生成不同的shift
            random_shift = random.randint(args.min_shift, args.max_shift)
            
            print(f"\n{'='*60}")
            print(f"Processing speaker: {speaker}")
            print(f"Random shift: {random_shift}")
            
            speaker_demo_dir = os.path.join(args.save_demo_dir, speaker)
            args_copy = argparse.Namespace(**vars(args))
            args_copy.save_demo_dir = speaker_demo_dir
            args_copy.demo_shift_frames = random_shift  # 使用随机shift
            
            try:
                export_demo(args_copy, extractor, model, device, video_path)
            except Exception as e:
                print(f"Error processing {speaker}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"All demos saved to {args.save_demo_dir}/")


# def main():
#     args = parse_args()
#     set_seed(args.seed)
#     device = get_device()
#     print(f"Using device: {device}")

#     # Load detector
#     model, saved_cfg = load_detector(args.detector_checkpoint, device)
#     print(f"Loaded detector from {args.detector_checkpoint}")

#     cfg = DetectorConfig(
#         sample_rate=saved_cfg.get("sample_rate", 16000),
#         n_mfcc=saved_cfg.get("n_mfcc", 20),
#         max_shift_frames=saved_cfg.get("max_shift_frames", 10),
#     )

#     speakers = args.speakers or sorted([d for d in os.listdir(args.data_path) if d.startswith("s")])
#     base_dataset = GridDataset(args.data_path, speakers, img_width=cfg.img_width, img_height=cfg.img_height, max_video_length=cfg.max_video_length)

#     lipnet = load_lipnet(args.checkpoint, len(base_dataset.vocab), device)
#     extractor = FeatureExtractor(base_dataset, lipnet, device, cfg)

#     # 如果指定了单个视频，只处理这个视频
#     if args.demo_video:
#         video_path = args.demo_video
#         export_demo(args, extractor, model, device, video_path)
#     else:
#         # 每个speaker随机选一个视频
#         videos_by_speaker = {}
#         for video_path, _ in base_dataset.samples:
#             # 从路径中提取speaker名字，例如 ./data/s1_processed/bbaf2n.mpg -> s1_processed
#             speaker = os.path.basename(os.path.dirname(video_path))
#             if speaker not in videos_by_speaker:
#                 videos_by_speaker[speaker] = []
#             videos_by_speaker[speaker].append(video_path)
        
#         print(f"Found {len(videos_by_speaker)} speakers")
        
#         # 每个speaker随机选一个
#         for speaker, videos in videos_by_speaker.items():
#             video_path = random.choice(videos)
#             print(f"\n{'='*60}")
#             print(f"Processing speaker: {speaker}")
            
#             # 为每个speaker创建独立的输出目录
#             speaker_demo_dir = os.path.join(args.save_demo_dir, speaker)
#             args_copy = argparse.Namespace(**vars(args))
#             args_copy.save_demo_dir = speaker_demo_dir
            
#             try:
#                 export_demo(args_copy, extractor, model, device, video_path)
#             except Exception as e:
#                 print(f"Error processing {speaker}: {e}")
#                 continue
        
#         print(f"\n{'='*60}")
#         print(f"All demos saved to {args.save_demo_dir}/")

# def main():
#     args = parse_args()
#     set_seed(args.seed)
#     device = get_device()
#     print(f"Using device: {device}")

#     # Load detector
#     model, saved_cfg = load_detector(args.detector_checkpoint, device)
#     print(f"Loaded detector from {args.detector_checkpoint}")

#     cfg = DetectorConfig(
#         sample_rate=saved_cfg.get("sample_rate", 16000),
#         n_mfcc=saved_cfg.get("n_mfcc", 20),
#         max_shift_frames=saved_cfg.get("max_shift_frames", 10),
#     )

#     speakers = args.speakers or sorted([d for d in os.listdir(args.data_path) if d.startswith("s")])
#     base_dataset = GridDataset(args.data_path, speakers, img_width=cfg.img_width, img_height=cfg.img_height, max_video_length=cfg.max_video_length)

#     lipnet = load_lipnet(args.checkpoint, len(base_dataset.vocab), device)
#     extractor = FeatureExtractor(base_dataset, lipnet, device, cfg)

#     # Get video
#     if args.demo_video:
#         video_path = args.demo_video
#     else:
#         videos = [v for v, _ in base_dataset.samples]
#         if not videos:
#             raise RuntimeError("No videos found")
#         video_path = random.choice(videos)

#     export_demo(args, extractor, model, device, video_path)


if __name__ == "__main__":
    main()