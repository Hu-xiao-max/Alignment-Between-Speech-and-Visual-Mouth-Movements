import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# ==================== Model Definition ====================

class LipNet(nn.Module):
    """LipNet model implementation in PyTorch"""
    
    def __init__(self, vocab_size: int, hidden_dim: int = 256, dropout_rate: float = 0.5):
        """
        Initialize LipNet model
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_dim: Hidden dimension for GRU layers
            dropout_rate: Dropout probability
        """
        super(LipNet, self).__init__()
        
        # 3D Convolutional layers for spatiotemporal feature extraction
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.dropout1 = nn.Dropout3d(dropout_rate)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.dropout2 = nn.Dropout3d(dropout_rate)
        
        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.dropout3 = nn.Dropout3d(dropout_rate)
        
        # Calculate the flattened dimension after convolutions
        # This depends on input size and pooling operations
        self.conv_output_dim = self._calculate_conv_output_dim()
        
        # Bidirectional GRU layers for sequence modeling
        self.gru1 = nn.GRU(self.conv_output_dim, hidden_dim, 
                           batch_first=True, bidirectional=True)
        self.dropout_gru1 = nn.Dropout(dropout_rate)
        
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, 
                           batch_first=True, bidirectional=True)
        self.dropout_gru2 = nn.Dropout(dropout_rate)
        
        # Output layer for CTC
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def _calculate_conv_output_dim(self):
        """Calculate the output dimension after convolutional layers"""
        # Assuming input shape: (1, 1, 75, 50, 100) - (batch, channel, time, height, width)
        # After 3 pooling layers with (1, 2, 2): height and width are divided by 8
        # Height: 50 / 8 = 6, Width: 100 / 8 = 12
        return 96 * 6 * 12  # channels * height * width
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels=1, time, height, width)
        Returns:
            Output tensor of shape (batch_size, time, vocab_size)
        """
        # 3D Convolution blocks
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Reshape for GRU: (batch, channels, time, h, w) -> (batch, time, features)
        batch_size, channels, time_steps, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, time, channels, h, w)
        x = x.view(batch_size, time_steps, -1)  # (batch, time, channels*h*w)
        
        # Bidirectional GRU layers
        x, _ = self.gru1(x)
        x = self.dropout_gru1(x)
        
        x, _ = self.gru2(x)
        x = self.dropout_gru2(x)
        
        # Output layer
        x = self.fc(x)
        
        # Apply log_softmax for CTC loss
        x = F.log_softmax(x, dim=-1)
        
        return x

# ==================== Training Functions ====================

class Trainer:
    """Trainer class for LipNet model"""
    
    def __init__(self, model, device, learning_rate=1e-4):
        """
        Initialize trainer
        
        Args:
            model: LipNet model
            device: Device to run on (cuda/cpu)
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.train_losses = []
        self.val_losses = []
        
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
            outputs = outputs.permute(1, 0, 2)
            
            # Input lengths (all sequences have same length after padding)
            input_lengths = torch.full((videos.size(0),), outputs.size(0), dtype=torch.long)
            
            loss = self.ctc_loss(outputs, labels, input_lengths, label_lengths)
            
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
                input_lengths = torch.full((videos.size(0),), outputs.size(0), dtype=torch.long)
                
                loss = self.ctc_loss(outputs, labels, input_lengths, label_lengths)
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, epochs):
        """
        Train the model for multiple epochs
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
        """
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, f'lipnet_checkpoint_epoch_{epoch+1}.pth')
                print(f"Saved checkpoint: epoch_{epoch+1}")
        
        # Save final model
        torch.save(self.model.state_dict(), 'lipnet_final.pth')
        print("\nTraining completed! Model saved.")
        
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.show()

# ==================== Evaluation Functions ====================

def decode_prediction(outputs, dataset, blank_index=0):
    """
    Decode CTC outputs to text using greedy decoding
    
    Args:
        outputs: Model outputs (log probabilities)
        dataset: Dataset object with vocabulary
        blank_index: Index for CTC blank label
    Returns:
        Decoded text string
    """
    # Get most likely characters
    _, predicted = torch.max(outputs, dim=-1)
    predicted = predicted.cpu().numpy()
    
    # Remove duplicates and blanks
    decoded = []
    prev_char = blank_index
    
    for char in predicted:
        if char != prev_char and char != blank_index:
            decoded.append(char)
        prev_char = char
    
    # Convert indices to characters
    text = ''.join([dataset.idx_to_char.get(idx, '') for idx in decoded 
                    if idx in dataset.idx_to_char])
    
    return text

def evaluate_model(model, test_loader, dataset, device, num_samples=5):
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained model
        test_loader: Test data loader
        dataset: Dataset object
        device: Device to run on
        num_samples: Number of samples to display
    """
    model.eval()
    model.to(device)
    
    print("\nModel Evaluation:")
    print("-" * 50)
    
    with torch.no_grad():
        for i, (videos, labels, label_lengths) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(videos)
            
            # Process each sample in the batch
            for j in range(videos.size(0)):
                if i * test_loader.batch_size + j >= num_samples:
                    break
                
                # Decode prediction
                predicted_text = decode_prediction(outputs[j], dataset)
                
                # Decode true label
                true_label = labels[j][:label_lengths[j]]
                true_text = ''.join([dataset.idx_to_char.get(idx.item(), '') 
                                   for idx in true_label if idx.item() != 0])
                
                print(f"\nSample {i * test_loader.batch_size + j + 1}:")
                print(f"True text: {true_text}")
                print(f"Predicted text: {predicted_text}")
                
                # Calculate character accuracy (simple version)
                correct_chars = sum(1 for a, b in zip(true_text, predicted_text) if a == b)
                accuracy = correct_chars / max(len(true_text), 1) * 100
                print(f"Character accuracy: {accuracy:.2f}%")

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
    
    # First, let's examine the structure of one directory in detail
    for item in items[:3]:  # Check first 3 directories
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            print(f"\n  Examining: {item}/")
            subdirs = os.listdir(item_path)
            print(f"    Contents: {subdirs[:10]}")
            
            # Check what's inside this directory
            for subdir in subdirs[:5]:
                subdir_path = os.path.join(item_path, subdir)
                if os.path.isdir(subdir_path):
                    files = os.listdir(subdir_path)[:3]
                    print(f"      {subdir}/: {files}")
                elif subdir.endswith(('.mpg', '.mp4', '.avi', '.npy', '.txt', '.align')):
                    print(f"      {subdir} (file)")
            
            # Check for different possible structures
            # Structure 1: video/ and align/ subdirectories
            video_path = os.path.join(item_path, 'video')
            align_path = os.path.join(item_path, 'align')
            
            # Structure 2: Direct video and text files
            video_files = [f for f in os.listdir(item_path) 
                          if f.endswith(('.mpg', '.mp4', '.avi', '.npy'))]
            text_files = [f for f in os.listdir(item_path) 
                         if f.endswith(('.txt', '.align'))]
            
            if os.path.exists(video_path) and os.path.exists(align_path):
                num_videos = len([f for f in os.listdir(video_path) 
                                if f.endswith(('.mpg', '.mp4', '.avi'))])
                print(f"    ✓ Standard structure: video/ and align/ folders with {num_videos} videos")
                speakers.append(item)
            elif video_files and text_files:
                print(f"    ✓ Flat structure: {len(video_files)} video files, {len(text_files)} text files")
                speakers.append(item)
            elif video_files:
                print(f"    ⚠ Found {len(video_files)} video files but no text files")
    
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

def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    CONFIG = {
        'data_path': './data',  # Update this path
        'train_speakers': None,  # Will be set dynamically
        'val_speakers': None,  # Will be set dynamically
        'test_speakers': None,  # Will be set dynamically
        'batch_size': 8,
        'epochs': 2,  # Reduced for quick testing, increase to 50 for real training
        'learning_rate': 1e-4,
        'img_width': 100,
        'img_height': 50,
        'max_video_length': 75,
        'hidden_dim': 256,
        'dropout_rate': 0.5,
        'quick_test': False,  # Set to True to skip training and only test data loading
    }
    
    # First, check the data structure
    available_speakers = check_data_structure(CONFIG['data_path'])
    
    if not available_speakers:
        print("\nERROR: No valid speaker directories found!")
        print("Expected structure:")
        print("  data/")
        print("    ├── s1/")
        print("    │   ├── video/")
        print("    │   │   └── *.mpg files")
        print("    │   └── align/")
        print("    │       └── *.align files")
        print("    ├── s2/")
        print("    └── ...")
        print("\nPlease check your data path and structure.")
        return
    
    print(f"\nFound {len(available_speakers)} valid speakers: {available_speakers}")
    
    # Automatically split speakers for train/val/test
    if len(available_speakers) < 3:
        print(f"ERROR: Need at least 3 speakers, but found only {len(available_speakers)}")
        return
    
    # Split speakers: 70% train, 15% val, 15% test
    num_speakers = len(available_speakers)
    num_train = max(1, int(num_speakers * 0.7))
    num_val = max(1, int(num_speakers * 0.15))
    
    CONFIG['train_speakers'] = available_speakers[:num_train]
    CONFIG['val_speakers'] = available_speakers[num_train:num_train + num_val]
    CONFIG['test_speakers'] = available_speakers[num_train + num_val:]
    
    # If test set is empty, use last validation speaker
    if not CONFIG['test_speakers'] and len(CONFIG['val_speakers']) > 1:
        CONFIG['test_speakers'] = [CONFIG['val_speakers'][-1]]
        CONFIG['val_speakers'] = CONFIG['val_speakers'][:-1]
    elif not CONFIG['test_speakers']:
        CONFIG['test_speakers'] = CONFIG['val_speakers']
    
    print(f"\nSpeaker split:")
    print(f"  Training: {CONFIG['train_speakers']}")
    print(f"  Validation: {CONFIG['val_speakers']}")
    print(f"  Test: {CONFIG['test_speakers']}")
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = GridDataset(
        CONFIG['data_path'], 
        CONFIG['train_speakers'],
        img_width=CONFIG['img_width'],
        img_height=CONFIG['img_height'],
        max_video_length=CONFIG['max_video_length']
    )
    
    print("Loading validation dataset...")
    val_dataset = GridDataset(
        CONFIG['data_path'],
        CONFIG['val_speakers'],
        img_width=CONFIG['img_width'],
        img_height=CONFIG['img_height'],
        max_video_length=CONFIG['max_video_length']
    )
    
    print("Loading test dataset...")
    test_dataset = GridDataset(
        CONFIG['data_path'],
        CONFIG['test_speakers'],
        img_width=CONFIG['img_width'],
        img_height=CONFIG['img_height'],
        max_video_length=CONFIG['max_video_length']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    if len(train_dataset) == 0:
        print("\nERROR: Training dataset is empty!")
        print("Possible reasons:")
        print("1. The data path is incorrect")
        print("2. The speakers don't have video/align subdirectories")
        print("3. No .mpg files found in video directories")
        print("\nExample of expected file structure:")
        print("  data/speaker_id/video/file.mpg")
        print("  data/speaker_id/align/file.align")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize model
    vocab_size = len(train_dataset.vocab)
    model = LipNet(
        vocab_size=vocab_size,
        hidden_dim=CONFIG['hidden_dim'],
        dropout_rate=CONFIG['dropout_rate']
    )
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Quick test mode - skip training
    if CONFIG.get('quick_test', False):
        print("\n=== QUICK TEST MODE - Skipping training ===")
        print("Testing data loading and model initialization only...")
        
        # Test forward pass with a single batch
        for batch_idx, (videos, labels, label_lengths) in enumerate(train_loader):
            if batch_idx == 0:
                videos = videos.to(device)
                print(f"Input shape: {videos.shape}")
                with torch.no_grad():
                    outputs = model(videos)
                print(f"Output shape: {outputs.shape}")
                print("✓ Model forward pass successful!")
                break
    else:
        # Initialize trainer
        trainer = Trainer(model, device, learning_rate=CONFIG['learning_rate'])
        
        # Train model
        trainer.train(train_loader, val_loader, CONFIG['epochs'])
        
        # Plot training history
        trainer.plot_losses()
    
    # If no test speakers available, skip test evaluation
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        print("\nEvaluating on test set...")
        evaluate_model(model, test_loader, test_dataset, device, num_samples=10)
    else:
        print("\nNo test data available, skipping test evaluation.")
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()