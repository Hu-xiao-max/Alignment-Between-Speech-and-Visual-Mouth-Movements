import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        
            if self.device.type == 'mps':
                loss = self.ctc_loss(outputs.cpu(), labels.cpu(), input_lengths.cpu(), label_lengths.cpu())
            else:
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
                
                # Handle MPS limitation for CTC loss
                if self.device.type == 'mps':
                    loss = self.ctc_loss(outputs.cpu(), labels.cpu(), input_lengths.cpu(), label_lengths.cpu())
                else:
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
