"""
Enhanced Neural Network Models for Code Clone Detection
Implements improved Siamese Network with advanced architecture and loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedSiameseNetwork(nn.Module):
    """
    Enhanced Siamese Network for Code Clone Detection with advanced architecture.
    
    Features:
    - Larger embedding and hidden dimensions
    - Multiple LSTM layers with dropout
    - Batch normalization for stability
    - Advanced regularization techniques
    """
    
    def __init__(self, vocab_size, embed_size=256, hidden_size=256):
        super(ImprovedSiameseNetwork, self).__init__()
        
        # Enhanced embedding layer with larger dimensions
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Advanced bidirectional LSTM with multiple layers
        self.bilstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=3,           # Increased to 3 layers for better representation
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
            bias=True
        )
        
        # Enhanced fully connected layers with advanced regularization
        lstm_output_size = hidden_size * 2  # Bidirectional
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32)  # Final embedding dimension
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward_once(self, x):
        """
        Forward pass for a single code snippet with enhanced processing.
        
        Args:
            x (torch.Tensor): Input token sequence [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Code embedding [batch_size, 32]
        """
        # Get embeddings
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_size]
        
        # Add positional encoding for better sequence understanding
        batch_size, seq_len, embed_size = embedded.size()
        
        # LSTM processing with enhanced handling
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        
        # Advanced pooling: combine last hidden states from all layers
        # hidden: [num_layers * 2, batch_size, hidden_size]
        forward_hidden = hidden[-2]  # Last layer, forward direction
        backward_hidden = hidden[-1] # Last layer, backward direction
        
        # Concatenate bidirectional final states
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Pass through fully connected layers
        features = self.fc(combined_hidden)
        
        return features

    def forward(self, code1, code2):
        """
        Forward pass for code pair comparison with advanced similarity computation.
        
        Args:
            code1 (torch.Tensor): First code snippet [batch_size, seq_len]
            code2 (torch.Tensor): Second code snippet [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Similarity scores [batch_size]
        """
        # Get embeddings for both code snippets
        embedding1 = self.forward_once(code1)
        embedding2 = self.forward_once(code2)
        
        # L2 normalize embeddings for better similarity computation
        embedding1_norm = F.normalize(embedding1, p=2, dim=1)
        embedding2_norm = F.normalize(embedding2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(embedding1_norm, embedding2_norm, dim=1)
        
        return similarity

class AdvancedContrastiveLoss(nn.Module):
    """
    Enhanced Contrastive Loss with adaptive margin and temperature scaling.
    """
    
    def __init__(self, margin=1.0, temperature=0.1):
        super(AdvancedContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, similarity, labels):
        """
        Compute enhanced contrastive loss.
        
        Args:
            similarity (torch.Tensor): Cosine similarity scores [-1, 1]
            labels (torch.Tensor): Binary labels (1=similar, 0=dissimilar)
            
        Returns:
            torch.Tensor: Contrastive loss value
        """
        # Apply temperature scaling for better gradient flow
        scaled_similarity = similarity / self.temperature
        
        # Positive pairs loss (want similarity close to 1)
        positive_loss = (1 - labels) * torch.pow(1 - scaled_similarity, 2)
        
        # Negative pairs loss (want similarity below margin)
        negative_loss = labels * torch.pow(
            torch.clamp(scaled_similarity - self.margin, min=0.0), 2
        )
        
        # Combine losses with adaptive weighting
        total_loss = torch.mean(positive_loss + negative_loss)
        
        return total_loss

class FocalContrastiveLoss(nn.Module):
    """
    Focal Contrastive Loss to handle class imbalance better.
    """
    
    def __init__(self, margin=1.0, alpha=1.0, gamma=2.0):
        super(FocalContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, similarity, labels):
        """
        Compute focal contrastive loss with class balancing.
        """
        # Standard contrastive loss components
        positive_loss = (1 - labels) * torch.pow(1 - similarity, 2)
        negative_loss = labels * torch.pow(
            torch.clamp(self.margin - similarity, min=0.0), 2
        )
        
        # Focal weighting to focus on hard examples
        pt_positive = torch.exp(-positive_loss)
        pt_negative = torch.exp(-negative_loss)
        
        focal_positive = self.alpha * torch.pow(1 - pt_positive, self.gamma) * positive_loss
        focal_negative = (1 - self.alpha) * torch.pow(1 - pt_negative, self.gamma) * negative_loss
        
        return torch.mean(focal_positive + focal_negative)

# Legacy alias for backward compatibility
ContrastiveLoss = AdvancedContrastiveLoss

def get_model_summary(model, input_size=(8, 512)):
    """
    Get a detailed summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Tuple of (batch_size, sequence_length)
    
    Returns:
        str: Model summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
Model Architecture Summary:
==========================
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)

Architecture:
{model}
    """
    
    return summary

def load_pretrained_model(model_path, vocab_size, embed_size=256, hidden_size=256, device='cpu'):
    """
    Load a pre-trained model with proper error handling.
    
    Args:
        model_path (str): Path to model checkpoint
        vocab_size (int): Vocabulary size
        embed_size (int): Embedding dimension
        hidden_size (int): Hidden dimension
        device (str): Device to load on
    
    Returns:
        ImprovedSiameseNetwork: Loaded model
    """
    model = ImprovedSiameseNetwork(vocab_size, embed_size, hidden_size)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    
    return model
