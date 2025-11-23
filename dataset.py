import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
import random

# ==================== Data Processing ====================

class GridDataset(Dataset):
    """GRID Corpus Dataset for PyTorch"""
    
    def __init__(self, data_path: str, speakers: List[str], 
                 img_width: int = 100, img_height: int = 50,
                 max_video_length: int = 75, transform=None):
        """
        Initialize the GRID dataset
        
        Args:
            data_path: Root directory path of the dataset
            speakers: List of speaker IDs
            img_width: Width of the processed frames
            img_height: Height of the processed frames
            max_video_length: Maximum number of frames in a video
            transform: Optional transform to be applied on frames
        """
        self.data_path = data_path
        self.speakers = speakers
        self.img_width = img_width
        self.img_height = img_height
        self.max_video_length = max_video_length
        self.transform = transform
        self.vocab = self._create_vocab()
        self.samples = self._load_file_paths()
        
    def _create_vocab(self) -> dict:
        """Create character to index mapping dictionary"""
        # Character set in GRID dataset
        characters = list("abcdefghijklmnopqrstuvwxyz0123456789 ")
        char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}  # Reserve 0 for blank
        char_to_idx['<blank>'] = 0  # CTC blank label
        char_to_idx['<pad>'] = len(characters) + 1  # Padding token
        self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        return char_to_idx
    
    def _load_file_paths(self) -> List[Tuple[str, str]]:
        """Load all video and align file paths"""
        samples = []
        
        for speaker in self.speakers:
            speaker_path = os.path.join(self.data_path, speaker)
            if not os.path.exists(speaker_path):
                print(f"Warning: Speaker path {speaker_path} does not exist")
                continue
            
            # Check for standard structure (video/ and align/ subdirectories)
            video_dir = os.path.join(speaker_path, 'video')
            align_dir = os.path.join(speaker_path, 'align')
            
            if os.path.exists(video_dir) and os.path.exists(align_dir):
                # Standard GRID structure
                video_formats = ['.mpg', '.mp4', '.avi', '.mov']
                video_files = []
                for fmt in video_formats:
                    video_files.extend([f for f in os.listdir(video_dir) if f.endswith(fmt)])
                
                for video_file in video_files:
                    video_path = os.path.join(video_dir, video_file)
                    base_name = os.path.splitext(video_file)[0]
                    
                    # Check for different align file formats
                    for ext in ['.align', '.txt']:
                        potential_align = base_name + ext
                        potential_path = os.path.join(align_dir, potential_align)
                        if os.path.exists(potential_path):
                            samples.append((video_path, potential_path))
                            break
            elif os.path.exists(align_dir):
                video_formats = ['.mpg', '.mp4', '.avi', '.mov']
                video_files = []
                for fmt in video_formats:
                    video_files.extend([f for f in os.listdir(speaker_path) if f.endswith(fmt)])
                
                for video_file in video_files:
                    video_path = os.path.join(speaker_path, video_file)
                    base_name = os.path.splitext(video_file)[0]
                    
                    # Check for different align file formats in align dir
                    for ext in ['.align', '.txt']:
                        potential_align = base_name + ext
                        potential_path = os.path.join(align_dir, potential_align)
                        if os.path.exists(potential_path):
                            samples.append((video_path, potential_path))
                            break
            else:
                # Check for flat structure (processed data)
                # Video files might be .npy (preprocessed) or original formats
                files = os.listdir(speaker_path)
                
                # Look for video files
                video_extensions = ['.mpg', '.mp4', '.avi', '.mov', '.npy']
                video_files = {}
                text_files = {}
                
                for file in files:
                    base_name = os.path.splitext(file)[0]
                    ext = os.path.splitext(file)[1]
                    
                    if ext in video_extensions:
                        video_files[base_name] = os.path.join(speaker_path, file)
                    elif ext in ['.txt', '.align']:
                        text_files[base_name] = os.path.join(speaker_path, file)
                
                # Match video and text files by base name
                for base_name in video_files:
                    if base_name in text_files:
                        samples.append((video_files[base_name], text_files[base_name]))
        
        if len(samples) == 0:
            print(f"Warning: No valid video-text pairs found for speakers {self.speakers}")
        else:
            print(f"Found {len(samples)} video-text pairs for speakers {self.speakers}")
        
        return samples
    
    def load_align_file(self, align_path: str) -> str:
        """
        Load alignment file to get text labels
        
        Args:
            align_path: Path to the alignment file
        Returns:
            Text label string
        """
        with open(align_path, 'r') as f:
            content = f.read()
        
        # Check if it's a simple text file (just the transcription)
        if not any(char.isdigit() for char in content.split('\n')[0]):
            # Simple text format - just return cleaned text
            text = content.strip().lower()
            return text
        
        # Standard GRID align format
        lines = content.strip().split('\n')
        
        # Extract words and concatenate into sentence
        words = []
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 3:
                    words.append(parts[2])
                elif len(parts) == 1:
                    # Sometimes the format is just the word
                    words.append(parts[0])
        
        # Replace 'sil' (silence) with space and clean up
        text = ' '.join(words).replace('sil', '').replace('sp', '').strip()
        return text.lower()
    
    def text_to_indices(self, text: str) -> torch.Tensor:
        """
        Convert text to index sequence
        
        Args:
            text: Input text string
        Returns:
            Tensor of indices
        """
        indices = [self.vocab.get(char, self.vocab['<pad>']) for char in text]
        return torch.LongTensor(indices)
    
    def process_video(self, video_path: str) -> torch.Tensor:
        """
        Process video file to extract mouth region frames
        
        Args:
            video_path: Path to the video file
        Returns:
            Tensor of processed video frames
        """
        # Check if it's a preprocessed numpy file
        if video_path.endswith('.npy'):
            frames = np.load(video_path)
            # Ensure proper shape and normalization
            if frames.max() > 1.0:
                frames = frames / 255.0
            
            # Resize if needed
            if frames.shape[1:] != (self.img_height, self.img_width):
                resized_frames = []
                for frame in frames:
                    resized = cv2.resize(frame, (self.img_width, self.img_height))
                    resized_frames.append(resized)
                frames = np.array(resized_frames)
        else:
            # Process video file
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                # Simple mouth region extraction (in practice, use dlib or mediapipe)
                # Assuming mouth is in the lower-middle part of the frame
                h, w = gray.shape
                mouth_region = gray[int(h*0.6):, int(w*0.3):int(w*0.7)]
                
                # Handle empty regions
                if mouth_region.size == 0:
                    mouth_region = gray  # Use full frame if region extraction fails
                
                # Resize to target dimensions
                mouth_resized = cv2.resize(mouth_region, (self.img_width, self.img_height))
                
                # Normalize pixel values
                mouth_normalized = mouth_resized / 255.0
                
                frames.append(mouth_normalized)
                
                # Limit maximum number of frames
                if len(frames) >= self.max_video_length:
                    break
            
            cap.release()
        
        if len(frames) == 0:
            print(f"Warning: No frames extracted from {video_path}")
            # Return dummy frames
            frames = np.zeros((self.max_video_length, self.img_height, self.img_width))
        else:
            frames = np.array(frames)
            
        # Pad or truncate to fixed length
        if len(frames) < self.max_video_length:
            # Pad with zeros
            padding = np.zeros((self.max_video_length - len(frames), self.img_height, self.img_width))
            frames = np.concatenate([frames, padding], axis=0)
        else:
            frames = frames[:self.max_video_length]
        
        # Convert to tensor: (T, H, W) -> (C=1, T, H, W)
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)
        
        return frames_tensor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, align_path = self.samples[idx]
        
        # Process video
        video_frames = self.process_video(video_path)
        
        # Get text label
        text_label = self.load_align_file(align_path)
        label_indices = self.text_to_indices(text_label)
        
        return video_frames, label_indices, len(label_indices)

def collate_fn(batch):
    """
    Custom collate function for batching variable-length sequences
    
    Args:
        batch: List of (video, label, label_length) tuples
    Returns:
        Batched tensors
    """
    videos, labels, label_lengths = zip(*batch)
    
    # Stack videos
    videos = torch.stack(videos, dim=0)
    
    # Pad labels to same length
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    # Convert lengths to tensor
    label_lengths = torch.LongTensor(label_lengths)
    
    return videos, labels, label_lengths
