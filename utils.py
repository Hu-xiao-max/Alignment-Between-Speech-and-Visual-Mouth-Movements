import os
import torch
import random
from typing import List, Tuple

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

# ==================== Helper Functions ====================

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
    for i, item in enumerate(items):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            if i < 3:
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
                if i < 3:
                    num_videos = len([f for f in os.listdir(video_path) 
                                    if f.endswith(('.mpg', '.mp4', '.avi'))])
                    print(f"    ✓ Standard structure: video/ and align/ folders with {num_videos} videos")
                speakers.append(item)
            elif os.path.exists(align_path) and video_files:
                # Structure 3: Videos in root, align in subfolder
                if i < 3:
                    num_aligns = len([f for f in os.listdir(align_path) if f.endswith(('.align', '.txt'))])
                    print(f"Mixed structure: {len(video_files)} videos in root, {num_aligns} alignments in align/ folder")
                speakers.append(item)
            elif video_files and text_files:
                if i < 3:
                    print(f"    ✓ Flat structure: {len(video_files)} video files, {len(text_files)} text files")
                speakers.append(item)
            elif video_files:
                if i < 3:
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
