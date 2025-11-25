import torch
import os
from torch.utils.data import DataLoader
from dataset import GridDataset, collate_fn
from model import LipNet
from utils import check_data_structure, evaluate_model

# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

def predict():
    """Load model and run predictions on test set"""

    model_path = 'lipnet_checkpoint_epoch_10.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run training first.")
        return

    # Configuration
    CONFIG = {
        'data_path': './data',
        'batch_size': 8,
        'img_width': 100,
        'img_height': 50,
        'max_video_length': 75,
        'hidden_dim': 256,
        'dropout_rate': 0.5,
    }
    
    # Check data structure
    available_speakers = check_data_structure(CONFIG['data_path'])
    if not available_speakers:
        print("No speakers found")
        return

    # Split speakers (same logic as main.py)
    num_speakers = len(available_speakers)
    num_train = max(1, int(num_speakers * 0.7))
    num_val = max(1, int(num_speakers * 0.15))
    
    test_speakers = available_speakers[num_train + num_val:]
    
    # If test set is empty, use last validation speaker
    if not test_speakers and len(available_speakers) > 1:
        test_speakers = [available_speakers[-1]]
    elif not test_speakers:
        test_speakers = available_speakers
        
    print(f"Test speakers: {test_speakers}")
    
    # Create test dataset
    print("Loading test dataset")
    test_dataset = GridDataset(
        CONFIG['data_path'],
        test_speakers,
        img_width=CONFIG['img_width'],
        img_height=CONFIG['img_height'],
        max_video_length=CONFIG['max_video_length']
    )
    
    if len(test_dataset) == 0:
        print("Test dataset is empty.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize model
    vocab_size = len(test_dataset.vocab)
    model = LipNet(
        vocab_size=vocab_size,
        hidden_dim=CONFIG['hidden_dim'],
        dropout_rate=CONFIG['dropout_rate']
    )
    
    # Load weights
    print(f"Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Handle if checkpoint is full state dict or just model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate
    evaluate_model(model, test_loader, test_dataset, device, num_samples=10)

if __name__ == "__main__":
    predict()
