import torch
from torch.utils.data import DataLoader
from dataset import GridDataset, collate_fn
from model import LipNet
from trainer import Trainer
from utils import check_data_structure, evaluate_model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== Main Program ====================

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
