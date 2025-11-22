import torch
import torch.nn as nn
import torch.nn.functional as F

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
