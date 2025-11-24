import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import random
from tqdm import tqdm
import json
import logging
from datetime import datetime
import argparse
try:
    import editdistance  # pip install editdistance
except ImportError:
    # Fallback implementation if editdistance is not installed
    def editdistance_eval(s1, s2):
        if len(s1) < len(s2):
            return editdistance_eval(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    class editdistance:
        @staticmethod
        def eval(s1, s2):
            return editdistance_eval(s1, s2)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ==================== Data Augmentation ====================

class VideoAugmentation:
    """Data augmentation for lip reading videos"""

    def __init__(self,
                 horizontal_flip: bool = True,
                 random_crop: bool = True,
                 time_mask: bool = True,
                 gaussian_noise: bool = True,
                 brightness_contrast: bool = True):
        self.horizontal_flip = horizontal_flip
        self.random_crop = random_crop
        self.time_mask = time_mask
        self.gaussian_noise = gaussian_noise
        self.brightness_contrast = brightness_contrast

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to video frames

        Args:
            frames: numpy array of shape (T, H, W)
        Returns:
            Augmented frames
        """
        # Horizontal flip (50% probability)
        if self.horizontal_flip and random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()

        # Random brightness/contrast adjustment
        if self.brightness_contrast and random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-0.1, 0.1)  # brightness
            frames = np.clip(frames * alpha + beta, 0, 1)

        # Add Gaussian noise
        if self.gaussian_noise and random.random() > 0.5:
            noise = np.random.normal(0, 0.02, frames.shape)
            frames = np.clip(frames + noise, 0, 1)

        # Time masking (mask random frames)
        if self.time_mask and random.random() > 0.5:
            T = frames.shape[0]
            mask_length = random.randint(1, max(1, T // 10))
            mask_start = random.randint(0, T - mask_length)
            frames[mask_start:mask_start + mask_length] = 0

        return frames.astype(np.float32)


# ==================== Data Processing ====================

def decode_grid_filename(filename: str) -> str:
    """
    Decode GRID corpus filename into the spoken phrase.

    GRID filename pattern: [command][color][preposition][letter][digit][adverb]
    Example: 'bbaj1s' -> 'bin blue at j one soon'

    Args:
        filename: Video filename (without extension)
    Returns:
        Decoded text phrase
    """
    # GRID corpus encoding
    commands = {'b': 'bin', 'l': 'lay', 'p': 'place', 's': 'set'}
    colors = {'b': 'blue', 'g': 'green', 'r': 'red', 'w': 'white'}
    prepositions = {'a': 'at', 'b': 'by', 'i': 'in', 'w': 'with'}
    digits = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        'z': 'zero'
    }
    adverbs = {'a': 'again', 'n': 'now', 'p': 'please', 's': 'soon'}

    # Remove extension if present
    base = os.path.splitext(filename)[0]

    if len(base) < 6:
        return ""

    try:
        command = commands.get(base[0].lower(), '')
        color = colors.get(base[1].lower(), '')
        prep = prepositions.get(base[2].lower(), '')
        letter = base[3].lower()
        digit = digits.get(base[4], base[4])
        adverb = adverbs.get(base[5].lower(), '')

        if all([command, color, prep, letter, digit, adverb]):
            return f"{command} {color} {prep} {letter} {digit} {adverb}"
    except (IndexError, KeyError):
        pass

    return ""


class GridDataset(Dataset):
    """GRID Corpus Dataset for PyTorch"""

    def __init__(self, data_path: str, speakers: List[str],
                 img_width: int = 128, img_height: int = 64,
                 max_video_length: int = 75, transform=None,
                 augmentation: Optional[VideoAugmentation] = None,
                 is_training: bool = True,
                 use_filename_as_label: bool = True):
        """
        Initialize the GRID dataset

        Args:
            data_path: Root directory path of the dataset
            speakers: List of speaker IDs
            img_width: Width of the processed frames
            img_height: Height of the processed frames
            max_video_length: Maximum number of frames in a video
            transform: Optional transform to be applied on frames
            augmentation: Optional video augmentation for training
            is_training: Whether this is a training dataset
            use_filename_as_label: If True, decode labels from GRID filenames
        """
        self.data_path = data_path
        self.speakers = speakers
        self.img_width = img_width
        self.img_height = img_height
        self.max_video_length = max_video_length
        self.transform = transform
        self.augmentation = augmentation
        self.is_training = is_training
        self.use_filename_as_label = use_filename_as_label
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

                    if self.use_filename_as_label:
                        # Decode label from filename
                        label = decode_grid_filename(base_name)
                        if label:
                            samples.append((video_path, label))
                    else:
                        # Check for different align file formats
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

                # Load samples
                for base_name, video_path in video_files.items():
                    if self.use_filename_as_label:
                        # Decode label from filename
                        label = decode_grid_filename(base_name)
                        if label:
                            samples.append((video_path, label))
                    elif base_name in text_files:
                        # Use alignment file
                        samples.append((video_path, text_files[base_name]))

        if len(samples) == 0:
            print(f"Warning: No valid video-text pairs found for speakers {self.speakers}")
        else:
            mode = "filename-decoded" if self.use_filename_as_label else "file-based"
            print(f"Found {len(samples)} {mode} samples for speakers {self.speakers}")

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
                
                # Mouth region extraction for GRID corpus
                # The face is in upper portion, mouth is around 40-65% of frame height
                h, w = gray.shape
                mouth_region = gray[int(h*0.4):int(h*0.65), int(w*0.3):int(w*0.7)]
                
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

        # Apply augmentation if in training mode
        if self.is_training and self.augmentation is not None:
            frames = self.augmentation(frames)

        # Convert to tensor: (T, H, W) -> (C=1, T, H, W)
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)

        return frames_tensor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label_or_path = self.samples[idx]

        # Process video
        video_frames = self.process_video(video_path)

        # Get text label
        if self.use_filename_as_label:
            # Label is already decoded from filename
            text_label = label_or_path
        else:
            # Label is a path to alignment file
            text_label = self.load_align_file(label_or_path)

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

# ==================== Model Definition ====================

class TemporalAttention(nn.Module):
    """Temporal attention mechanism to focus on relevant frames"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch, time, features)
        weights = self.attention(x)  # (batch, time, 1)
        weights = F.softmax(weights, dim=1)
        # Return attended features but also preserve sequence
        # We use attention as a gate, not as aggregation
        return x * (1 + weights)  # Enhance attended features


class LipNet(nn.Module):
    """LipNet model with attention and better regularization"""

    def __init__(self, vocab_size: int, hidden_dim: int = 256, dropout_rate: float = 0.5):
        """
        Initialize LipNet model

        Args:
            vocab_size: Size of the vocabulary
            hidden_dim: Hidden dimension for GRU layers
            dropout_rate: Dropout probability
        """
        super(LipNet, self).__init__()

        # 3D Convolutional layers with batch normalization
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout1 = nn.Dropout3d(dropout_rate)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout2 = nn.Dropout3d(dropout_rate)

        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout3 = nn.Dropout3d(dropout_rate)

        # Calculate the flattened dimension after convolutions
        self.conv_output_dim = 96 * 4 * 8  # 3072

        # Bidirectional GRU layers
        self.gru1 = nn.GRU(self.conv_output_dim, hidden_dim,
                           batch_first=True, bidirectional=True)
        self.dropout_gru1 = nn.Dropout(dropout_rate)

        # Attention mechanism
        self.attention = TemporalAttention(hidden_dim)

        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim,
                           batch_first=True, bidirectional=True)
        self.dropout_gru2 = nn.Dropout(dropout_rate)

        # Output layer for CTC
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming for conv and orthogonal for GRU"""
        # Initialize conv layers with Kaiming normal
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                # Initialize GRU weights
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, channels=1, time, height, width)
        Returns:
            Output tensor of shape (batch_size, time, vocab_size)
        """
        # 3D Convolution blocks with batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Reshape for GRU: (batch, channels, time, h, w) -> (batch, time, features)
        batch_size, _, time_steps, _, _ = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, time, channels, h, w)
        x = x.view(batch_size, time_steps, -1)  # (batch, time, channels*h*w)

        # First GRU layer
        x, _ = self.gru1(x)
        x = self.dropout_gru1(x)

        # Apply temporal attention
        x = self.attention(x)

        # Second GRU layer
        x, _ = self.gru2(x)
        x = self.dropout_gru2(x)

        # Output layer
        x = self.fc(x)

        # Apply log_softmax for CTC loss
        x = F.log_softmax(x, dim=-1)

        return x

# ==================== Training Functions ====================

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


class Trainer:
    """Trainer class for LipNet model"""

    def __init__(self, model, device, learning_rate=1e-4,
                 scheduler_type: str = 'plateau',
                 early_stopping_patience: int = 15,
                 blank_penalty: float = 0.0):
        """
        Initialize trainer

        Args:
            model: LipNet model
            device: Device to run on (cuda/cpu)
            learning_rate: Learning rate for optimizer
            scheduler_type: Type of learning rate scheduler ('plateau', 'cosine', 'none')
            early_stopping_patience: Patience for early stopping
            blank_penalty: Penalty weight for predicting too many blanks (0.0 to disable)
        """
        self.model = model.to(device)
        self.device = device
        self.blank_penalty = blank_penalty
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Setup learning rate scheduler
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5,
                patience=5, verbose=True, min_lr=1e-7
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
            )
        else:
            self.scheduler = None

        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self, dataloader):
        """
        Train for one epoch

        Args:
            dataloader: Training data loader
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc='Training')
        for videos, labels, label_lengths in progress_bar:
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            label_lengths = label_lengths.to(self.device)

            # Forward pass
            outputs = self.model(videos)

            # Calculate CTC loss
            # outputs shape: (batch, time, vocab)
            # Need to transpose for CTC loss: (time, batch, vocab)
            outputs_ctc = outputs.permute(1, 0, 2)

            # Input lengths (all sequences have same length after padding)
            input_lengths = torch.full((videos.size(0),), outputs_ctc.size(0), dtype=torch.long, device='cpu')
            label_lengths_cpu = label_lengths.cpu()

            loss = self.ctc_loss(outputs_ctc, labels, input_lengths, label_lengths_cpu)

            # Diversity loss: penalize if all samples in batch have similar outputs
            # Encourage different predictions for different inputs
            if outputs.size(0) > 1:
                # Get predicted character distributions for each sample
                probs = torch.softmax(outputs, dim=-1)  # (batch, time, vocab)
                # Average over time to get sample-level distribution
                avg_probs = probs.mean(dim=1)  # (batch, vocab)
                # Calculate pairwise KL divergence to encourage diversity
                batch_mean = avg_probs.mean(dim=0, keepdim=True)  # (1, vocab)
                # Negative entropy of batch mean (encourage spread)
                diversity_loss = (batch_mean * torch.log(batch_mean + 1e-8)).sum()
                # Add small diversity term
                loss = loss + 0.1 * diversity_loss

            # Add blank penalty to prevent blank collapse
            if self.blank_penalty > 0:
                blank_probs = torch.softmax(outputs_ctc, dim=-1)[:, :, 0]  # (time, batch)
                blank_loss = blank_probs.mean() * self.blank_penalty
                loss = loss + blank_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches
    
    def validate(self, dataloader):
        """
        Validate the model
        
        Args:
            dataloader: Validation data loader
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Validation')
            for videos, labels, label_lengths in progress_bar:
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                label_lengths = label_lengths.to(self.device)
                
                # Forward pass
                outputs = self.model(videos)
                
                # Calculate CTC loss
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((videos.size(0),), outputs.size(0), dtype=torch.long, device='cpu')
                label_lengths_cpu = label_lengths.cpu()

                loss = self.ctc_loss(outputs, labels, input_lengths, label_lengths_cpu)
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, epochs, save_dir: str = 'checkpoints'):
        """
        Train the model for multiple epochs

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")

            # Training phase
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            logger.info(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, LR: {current_lr:.2e}")

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                best_path = os.path.join(save_dir, 'lipnet_best.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': self.best_val_loss,
                }, best_path)
                logger.info(f"New best model saved! Val loss: {val_loss:.4f}")

            # Early stopping check
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Save checkpoint periodically
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                }
                ckpt_path = os.path.join(save_dir, f'lipnet_checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint: epoch_{epoch+1}")

        # Load best model weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model weights")

        # Save final model
        final_path = os.path.join(save_dir, 'lipnet_final.pth')
        torch.save(self.model.state_dict(), final_path)
        logger.info("Training completed! Model saved.")

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss
        }
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
    def plot_losses(self, save_dir: str = 'checkpoints'):
        """Plot training and validation losses with learning rate"""
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot learning rate
        if self.learning_rates:
            ax2.plot(self.learning_rates, label='Learning Rate', color='green')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
        plt.show()

    @staticmethod
    def load_checkpoint(model, checkpoint_path: str, device, optimizer=None):
        """
        Load model from checkpoint

        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint file
            device: Device to load model to
            optimizer: Optional optimizer to load state
        Returns:
            Loaded model, optimizer (if provided), and checkpoint dict
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Best val loss: {checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'unknown'))}")

        return model, optimizer, checkpoint

# ==================== Evaluation Functions ====================

def calculate_cer(prediction: str, target: str) -> float:
    """
    Calculate Character Error Rate (CER)

    Args:
        prediction: Predicted text string
        target: Ground truth text string
    Returns:
        Character Error Rate (lower is better)
    """
    if len(target) == 0:
        return 1.0 if len(prediction) > 0 else 0.0
    return editdistance.eval(prediction, target) / len(target)


def calculate_wer(prediction: str, target: str) -> float:
    """
    Calculate Word Error Rate (WER)

    Args:
        prediction: Predicted text string
        target: Ground truth text string
    Returns:
        Word Error Rate (lower is better)
    """
    pred_words = prediction.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    return editdistance.eval(pred_words, target_words) / len(target_words)


def decode_prediction(outputs, dataset, blank_index=0, debug=False):
    """
    Decode CTC outputs to text using greedy decoding

    Args:
        outputs: Model outputs (log probabilities)
        dataset: Dataset object with vocabulary
        blank_index: Index for CTC blank label
        debug: Whether to print debug information
    Returns:
        Decoded text string
    """
    # Get most likely characters
    probs, predicted = torch.max(outputs, dim=-1)
    predicted = predicted.cpu().numpy()
    probs = probs.cpu().numpy()

    if debug:
        # Count how many predictions are blank vs non-blank
        blank_count = (predicted == blank_index).sum()
        non_blank_count = len(predicted) - blank_count
        unique_chars = np.unique(predicted)
        logger.info(f"  Debug: {blank_count} blanks, {non_blank_count} non-blanks out of {len(predicted)} timesteps")
        logger.info(f"  Debug: Unique predicted indices: {unique_chars[:20]}")
        # Show probability distribution for first few timesteps
        if len(outputs) > 0:
            sample_probs = torch.softmax(outputs[:5], dim=-1)
            max_probs, max_idx = torch.max(sample_probs, dim=-1)
            logger.info(f"  Debug: First 5 timestep max probs: {max_probs.cpu().numpy()}")
            logger.info(f"  Debug: First 5 timestep max indices: {max_idx.cpu().numpy()}")

    # Remove duplicates and blanks
    decoded = []
    prev_char = blank_index

    for char in predicted:
        if char != prev_char and char != blank_index:
            decoded.append(char)
        prev_char = char

    # Convert indices to characters
    text = ''.join([dataset.idx_to_char.get(idx, '') for idx in decoded
                    if idx in dataset.idx_to_char and idx != blank_index])

    return text


def beam_search_decode(outputs, dataset, beam_width: int = 10, blank_index: int = 0) -> str:
    """
    Decode CTC outputs using beam search

    Args:
        outputs: Model outputs (log probabilities) shape (T, vocab_size)
        dataset: Dataset object with vocabulary
        beam_width: Number of beams to keep
        blank_index: Index for CTC blank label
    Returns:
        Best decoded text string
    """
    T, vocab_size = outputs.shape
    outputs = outputs.cpu().numpy()

    # Initialize beams: (prefix, last_char, score)
    beams = [('', blank_index, 0.0)]

    for t in range(T):
        new_beams = {}

        for prefix, last_char, score in beams:
            for c in range(vocab_size):
                new_score = score + outputs[t, c]

                if c == blank_index:
                    # Blank: keep prefix unchanged
                    key = (prefix, blank_index)
                    if key not in new_beams or new_beams[key][2] < new_score:
                        new_beams[key] = (prefix, blank_index, new_score)
                elif c == last_char:
                    # Same as last non-blank: don't extend
                    key = (prefix, c)
                    if key not in new_beams or new_beams[key][2] < new_score:
                        new_beams[key] = (prefix, c, new_score)
                else:
                    # New character: extend prefix
                    char = dataset.idx_to_char.get(c, '')
                    new_prefix = prefix + char
                    key = (new_prefix, c)
                    if key not in new_beams or new_beams[key][2] < new_score:
                        new_beams[key] = (new_prefix, c, new_score)

        # Keep top beam_width beams
        beams = sorted(new_beams.values(), key=lambda x: x[2], reverse=True)[:beam_width]

    # Return best beam
    if beams:
        return beams[0][0]
    return ''

def evaluate_model(model, test_loader, dataset, device, num_samples: int = 5,
                   use_beam_search: bool = False, beam_width: int = 10) -> Dict:
    """
    Evaluate model performance on test data

    Args:
        model: Trained model
        test_loader: Test data loader
        dataset: Dataset object
        device: Device to run on
        num_samples: Number of samples to display (-1 for all)
        use_beam_search: Whether to use beam search decoding
        beam_width: Beam width for beam search
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)

    all_predictions = []
    all_targets = []
    total_cer = 0.0
    total_wer = 0.0
    sample_count = 0

    logger.info("\nModel Evaluation:")
    logger.info("-" * 50)

    with torch.no_grad():
        for i, (videos, labels, label_lengths) in enumerate(tqdm(test_loader, desc='Evaluating')):
            videos = videos.to(device)
            labels = labels.to(device)

            # Get predictions
            outputs = model(videos)

            # Process each sample in the batch
            for j in range(videos.size(0)):
                # Decode prediction
                # Enable debug for first sample to see what model is outputting
                debug_mode = (sample_count == 0)
                if use_beam_search:
                    predicted_text = beam_search_decode(outputs[j], dataset, beam_width)
                else:
                    predicted_text = decode_prediction(outputs[j], dataset, debug=debug_mode)

                # Decode true label
                true_label = labels[j][:label_lengths[j]]
                true_text = ''.join([dataset.idx_to_char.get(idx.item(), '')
                                   for idx in true_label if idx.item() != 0])

                # Calculate metrics
                cer = calculate_cer(predicted_text, true_text)
                wer = calculate_wer(predicted_text, true_text)

                total_cer += cer
                total_wer += wer
                sample_count += 1

                all_predictions.append(predicted_text)
                all_targets.append(true_text)

                # Display samples
                if num_samples == -1 or sample_count <= num_samples:
                    logger.info(f"\nSample {sample_count}:")
                    logger.info(f"  Target:     '{true_text}'")
                    logger.info(f"  Predicted:  '{predicted_text}'")
                    logger.info(f"  CER: {cer*100:.2f}%, WER: {wer*100:.2f}%")

    # Calculate average metrics
    avg_cer = total_cer / max(sample_count, 1)
    avg_wer = total_wer / max(sample_count, 1)

    # Calculate overall accuracy
    correct_predictions = sum(1 for p, t in zip(all_predictions, all_targets) if p == t)
    accuracy = correct_predictions / max(sample_count, 1)

    results = {
        'total_samples': sample_count,
        'avg_cer': avg_cer,
        'avg_wer': avg_wer,
        'accuracy': accuracy,
        'correct_predictions': correct_predictions,
    }

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS:")
    logger.info(f"  Total Samples: {sample_count}")
    logger.info(f"  Average CER: {avg_cer*100:.2f}%")
    logger.info(f"  Average WER: {avg_wer*100:.2f}%")
    logger.info(f"  Exact Match Accuracy: {accuracy*100:.2f}% ({correct_predictions}/{sample_count})")
    logger.info("=" * 50)

    return results


# ==================== Inference Pipeline ====================

class LipReadingInference:
    """Inference class for lip reading on new videos"""

    def __init__(self, model_path: str, device: str = 'cuda',
                 img_width: int = 100, img_height: int = 50,
                 max_video_length: int = 75):
        """
        Initialize inference pipeline

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            img_width: Width of processed frames
            img_height: Height of processed frames
            max_video_length: Maximum number of frames
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_width = img_width
        self.img_height = img_height
        self.max_video_length = max_video_length

        # Create vocabulary
        self.vocab, self.idx_to_char = self._create_vocab()

        # Load model
        self.model = self._load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")

    def _create_vocab(self) -> Tuple[dict, dict]:
        """Create character to index mapping"""
        characters = list("abcdefghijklmnopqrstuvwxyz0123456789 ")
        char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
        char_to_idx['<blank>'] = 0
        char_to_idx['<pad>'] = len(characters) + 1
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        return char_to_idx, idx_to_char

    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        vocab_size = len(self.vocab)
        model = LipNet(vocab_size=vocab_size)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """
        Preprocess video for inference

        Args:
            video_path: Path to video file
        Returns:
            Preprocessed video tensor
        """
        if video_path.endswith('.npy'):
            frames = np.load(video_path)
            if frames.max() > 1.0:
                frames = frames / 255.0
        else:
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

                # Extract mouth region for GRID corpus
                h, w = gray.shape
                mouth_region = gray[int(h*0.4):int(h*0.65), int(w*0.3):int(w*0.7)]

                if mouth_region.size == 0:
                    mouth_region = gray

                # Resize and normalize
                mouth_resized = cv2.resize(mouth_region, (self.img_width, self.img_height))
                mouth_normalized = mouth_resized / 255.0
                frames.append(mouth_normalized)

                if len(frames) >= self.max_video_length:
                    break

            cap.release()
            frames = np.array(frames) if frames else np.zeros((1, self.img_height, self.img_width))

        # Pad or truncate
        if len(frames) < self.max_video_length:
            padding = np.zeros((self.max_video_length - len(frames), self.img_height, self.img_width))
            frames = np.concatenate([frames, padding], axis=0)
        else:
            frames = frames[:self.max_video_length]

        # Convert to tensor: (T, H, W) -> (1, 1, T, H, W)
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0)
        return frames_tensor

    def predict(self, video_path: str, use_beam_search: bool = False,
                beam_width: int = 10) -> str:
        """
        Predict text from video

        Args:
            video_path: Path to video file
            use_beam_search: Whether to use beam search decoding
            beam_width: Beam width for beam search
        Returns:
            Predicted text
        """
        # Preprocess video
        video_tensor = self.preprocess_video(video_path).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(video_tensor)

        # Decode output
        if use_beam_search:
            # Create a simple object with idx_to_char for beam search
            class VocabObj:
                pass
            vocab_obj = VocabObj()
            vocab_obj.idx_to_char = self.idx_to_char
            predicted_text = beam_search_decode(outputs[0], vocab_obj, beam_width)
        else:
            # Greedy decoding
            _, predicted = torch.max(outputs[0], dim=-1)
            predicted = predicted.cpu().numpy()

            decoded = []
            prev_char = 0  # blank
            for char in predicted:
                if char != prev_char and char != 0:
                    decoded.append(char)
                prev_char = char

            predicted_text = ''.join([self.idx_to_char.get(idx, '') for idx in decoded
                                     if idx in self.idx_to_char])

        return predicted_text

    def predict_batch(self, video_paths: List[str], use_beam_search: bool = False) -> List[str]:
        """
        Predict text from multiple videos

        Args:
            video_paths: List of video file paths
            use_beam_search: Whether to use beam search decoding
        Returns:
            List of predicted texts
        """
        predictions = []
        for video_path in tqdm(video_paths, desc='Processing videos'):
            pred = self.predict(video_path, use_beam_search)
            predictions.append(pred)
        return predictions


def run_inference(model_path: str, video_path: str, use_beam_search: bool = False):
    """
    Run inference on a single video

    Args:
        model_path: Path to trained model
        video_path: Path to video file
        use_beam_search: Whether to use beam search decoding
    """
    inference = LipReadingInference(model_path)
    prediction = inference.predict(video_path, use_beam_search)

    logger.info(f"\nVideo: {video_path}")
    logger.info(f"Prediction: '{prediction}'")

    return prediction


# ==================== Main Program ====================

def find_alignment_files(data_path):
    """Search for alignment files in various possible locations"""
    print("\nSearching for alignment/text files...")
    
    # Check parent directory
    parent_dir = os.path.dirname(data_path)
    print(f"Checking parent directory: {parent_dir}")
    
    # Look for common alignment directory names
    possible_align_dirs = ['align', 'alignments', 'transcriptions', 'labels', 'text']
    
    for align_dir in possible_align_dirs:
        # Check in parent directory
        align_path = os.path.join(parent_dir, align_dir)
        if os.path.exists(align_path):
            print(f"  Found potential alignment directory: {align_path}")
            files = os.listdir(align_path)[:5]
            print(f"    Sample files: {files}")
    
    # Check if there's a separate align folder for each speaker
    for speaker_dir in os.listdir(data_path)[:3]:
        speaker_path = os.path.join(data_path, speaker_dir)
        if os.path.isdir(speaker_path):
            # Remove _processed suffix to find original speaker ID
            speaker_id = speaker_dir.replace('_processed', '')
            
            # Check various possible locations
            possible_paths = [
                os.path.join(parent_dir, 'align', speaker_id),
                os.path.join(parent_dir, speaker_id, 'align'),
                os.path.join(data_path, speaker_id, 'align'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"  Found alignment files at: {path}")
                    return True
    
    # Check if alignment files have different naming pattern
    sample_speaker = os.listdir(data_path)[0]
    sample_path = os.path.join(data_path, sample_speaker)
    video_files = [f for f in os.listdir(sample_path) if f.endswith('.mpg')]
    
    if video_files:
        sample_video = video_files[0]
        base_name = os.path.splitext(sample_video)[0]
        print(f"\nSample video: {sample_video}")
        print(f"Looking for matching text file with base name: {base_name}")
    
    return False

def create_dummy_alignments(data_path):
    """Create dummy alignment files for testing purposes"""
    print("\nNo alignment files found. Creating dummy alignments for testing...")
    
    # Common phrases in GRID corpus
    grid_phrases = [
        "bin blue at f nine please",
        "lay red at j two now",
        "place white by a four soon",
        "set green in x eight again",
        "bin blue at l three please",
        "lay red by r zero now",
        "place white at u five soon",
        "set green by b six again",
    ]
    
    created_count = 0
    for speaker_dir in os.listdir(data_path):
        speaker_path = os.path.join(data_path, speaker_dir)
        if os.path.isdir(speaker_path):
            video_files = [f for f in os.listdir(speaker_path) if f.endswith('.mpg')]
            
            for video_file in video_files[:10]:  # Create for first 10 videos per speaker
                base_name = os.path.splitext(video_file)[0]
                align_file = base_name + '.txt'
                align_path = os.path.join(speaker_path, align_file)
                
                # Random phrase from GRID corpus
                phrase = random.choice(grid_phrases)
                
                with open(align_path, 'w') as f:
                    f.write(phrase)
                
                created_count += 1
    
    print(f"Created {created_count} dummy alignment files for testing")
    return True

def check_data_structure(data_path):
    """Check and print the actual structure of the dataset"""
    print(f"\nChecking data structure at: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data path does not exist: {data_path}")
        return []
    
    # List all directories in data path
    items = sorted(os.listdir(data_path))
    speakers = []
    
    print(f"Found {len(items)} items in data directory:")
    
    # Examine all directories and collect valid speakers
    for item in items:  # Check all directories
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            # Check for different possible structures
            video_path = os.path.join(item_path, 'video')
            align_path = os.path.join(item_path, 'align')

            # Structure 2: Direct video and text files
            video_files = [f for f in os.listdir(item_path)
                          if f.endswith(('.mpg', '.mp4', '.avi', '.npy'))]

            if os.path.exists(video_path) and os.path.exists(align_path):
                speakers.append(item)
            elif video_files:
                # With use_filename_as_label=True, we can use any video files
                speakers.append(item)
    
    # Show all speaker directories found
    all_speaker_dirs = [d for d in items if os.path.isdir(os.path.join(data_path, d))]
    print(f"\nAll directories found: {all_speaker_dirs}")
    
    # Search for alignment files
    find_alignment_files(data_path)
    
    # If no speakers with valid structure, offer to create dummy alignments
    if not speakers:
        print("\nWARNING: No directories with both video and text files found.")
        response = input("\nDo you want to create dummy alignment files for testing? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            if create_dummy_alignments(data_path):
                # Re-check structure after creating dummy files
                for item in all_speaker_dirs:
                    item_path = os.path.join(data_path, item)
                    video_files = [f for f in os.listdir(item_path) 
                                  if f.endswith(('.mpg', '.mp4', '.avi', '.npy'))]
                    text_files = [f for f in os.listdir(item_path) 
                                 if f.endswith(('.txt', '.align'))]
                    if video_files and text_files:
                        speakers.append(item)
    
    return speakers if speakers else all_speaker_dirs

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LipNet: End-to-End Sentence-level Lipreading',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'inference'],
                       help='Running mode: train, test, or inference')

    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to dataset directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (for test/inference or resume training)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler type')
    parser.add_argument('--early_stopping', type=int, default=15,
                       help='Early stopping patience (0 to disable)')
    parser.add_argument('--blank_penalty', type=float, default=1.0,
                       help='Penalty for blank predictions to prevent CTC collapse (0.0 to disable)')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for GRU layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')

    # Data processing arguments
    parser.add_argument('--img_width', type=int, default=128,
                       help='Width of processed frames')
    parser.add_argument('--img_height', type=int, default=64,
                       help='Height of processed frames')
    parser.add_argument('--max_frames', type=int, default=75,
                       help='Maximum number of frames per video')

    # Augmentation arguments
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')

    # Inference arguments
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file for inference')
    parser.add_argument('--beam_search', action='store_true',
                       help='Use beam search decoding')
    parser.add_argument('--beam_width', type=int, default=10,
                       help='Beam width for beam search')

    # Output arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Test arguments
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode (skip training)')

    return parser.parse_args()


def main():
    """Main training and evaluation pipeline"""

    args = parse_args()

    logger.info("=" * 60)
    logger.info("LipNet Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {device}")

    # Handle inference mode separately
    if args.mode == 'inference':
        if args.checkpoint is None:
            logger.error("Checkpoint required for inference mode. Use --checkpoint")
            return
        if args.video is None:
            logger.error("Video path required for inference mode. Use --video")
            return

        prediction = run_inference(args.checkpoint, args.video, args.beam_search)
        logger.info(f"Prediction: {prediction}")
        return

    # First, check the data structure
    available_speakers = check_data_structure(args.data_path)

    if not available_speakers:
        logger.error("No valid speaker directories found!")
        logger.error("Expected structure:")
        logger.error("  data/")
        logger.error("    ├── s1/")
        logger.error("    │   ├── video/")
        logger.error("    │   │   └── *.mpg files")
        logger.error("    │   └── align/")
        logger.error("    │       └── *.align files")
        logger.error("    ├── s2/")
        logger.error("    └── ...")
        return

    logger.info(f"Found {len(available_speakers)} valid speakers: {available_speakers}")

    # Automatically split speakers for train/val/test
    if len(available_speakers) < 3:
        logger.error(f"Need at least 3 speakers, but found only {len(available_speakers)}")
        return

    # Split speakers: 70% train, 15% val, 15% test
    num_speakers = len(available_speakers)
    num_train = max(1, int(num_speakers * 0.7))
    num_val = max(1, int(num_speakers * 0.15))

    train_speakers = available_speakers[:num_train]
    val_speakers = available_speakers[num_train:num_train + num_val]
    test_speakers = available_speakers[num_train + num_val:]

    # If test set is empty, use last validation speaker
    if not test_speakers and len(val_speakers) > 1:
        test_speakers = [val_speakers[-1]]
        val_speakers = val_speakers[:-1]
    elif not test_speakers:
        test_speakers = val_speakers

    logger.info(f"Speaker split:")
    logger.info(f"  Training: {train_speakers}")
    logger.info(f"  Validation: {val_speakers}")
    logger.info(f"  Test: {test_speakers}")

    # Create augmentation (only for training)
    train_augmentation = None if args.no_augmentation else VideoAugmentation()

    # Create datasets
    logger.info("Loading training dataset...")
    train_dataset = GridDataset(
        args.data_path,
        train_speakers,
        img_width=args.img_width,
        img_height=args.img_height,
        max_video_length=args.max_frames,
        augmentation=train_augmentation,
        is_training=True
    )

    logger.info("Loading validation dataset...")
    val_dataset = GridDataset(
        args.data_path,
        val_speakers,
        img_width=args.img_width,
        img_height=args.img_height,
        max_video_length=args.max_frames,
        augmentation=None,
        is_training=False
    )

    logger.info("Loading test dataset...")
    test_dataset = GridDataset(
        args.data_path,
        test_speakers,
        img_width=args.img_width,
        img_height=args.img_height,
        max_video_length=args.max_frames,
        augmentation=None,
        is_training=False
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty!")
        logger.error("Possible reasons:")
        logger.error("1. The data path is incorrect")
        logger.error("2. The speakers don't have video/align subdirectories")
        logger.error("3. No .mpg files found in video directories")
        return

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize model
    vocab_size = len(train_dataset.vocab)
    model = LipNet(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout
    )

    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load checkpoint if provided
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Resumed from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded model weights")

    # Test mode - only evaluate
    if args.mode == 'test':
        model.to(device)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )

        logger.info("Running evaluation on test set...")
        results = evaluate_model(
            model, test_loader, test_dataset, device,
            num_samples=-1,  # Evaluate all samples
            use_beam_search=args.beam_search,
            beam_width=args.beam_width
        )

        # Save results
        with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        return

    # Quick test mode - skip training
    if args.quick_test:
        logger.info("=== QUICK TEST MODE - Skipping training ===")
        logger.info("Testing data loading and model initialization only...")

        model.to(device)
        # Test forward pass with a single batch
        for batch_idx, (videos, _, _) in enumerate(train_loader):
            if batch_idx == 0:
                videos = videos.to(device)
                logger.info(f"Input shape: {videos.shape}")
                with torch.no_grad():
                    outputs = model(videos)
                logger.info(f"Output shape: {outputs.shape}")
                logger.info("Model forward pass successful!")
                break
        return

    # Initialize trainer
    trainer = Trainer(
        model, device,
        learning_rate=args.learning_rate,
        scheduler_type=args.scheduler,
        early_stopping_patience=args.early_stopping if args.early_stopping > 0 else 999,
        blank_penalty=args.blank_penalty
    )

    # Train model
    trainer.train(train_loader, val_loader, args.epochs, save_dir=args.save_dir)

    # Plot training history
    trainer.plot_losses(save_dir=args.save_dir)

    # Evaluate on test set
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )

        logger.info("Evaluating on test set...")
        results = evaluate_model(
            model, test_loader, test_dataset, device,
            num_samples=20,
            use_beam_search=args.beam_search,
            beam_width=args.beam_width
        )

        # Save results
        with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    else:
        logger.info("No test data available, skipping test evaluation.")

    logger.info("Training and evaluation completed!")


if __name__ == "__main__":
    main()