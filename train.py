"""
Enhanced Training Module for Code Clone Detection
Implements advanced training with validation, early stopping, and checkpointing
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
import time
import os
from sklearn.metrics import f1_score, accuracy_score

from dataset import load_bigclonebench_data_robust
from models import ImprovedSiameseNetwork, AdvancedContrastiveLoss

class AdvancedTrainer:
    """
    Advanced trainer with validation, early stopping, and comprehensive metrics.
    """
    
    def __init__(self, model, criterion, optimizer, scheduler, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_f1 = 0.0
        self.best_model_path = None
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch, total_epochs):
        """Train for one epoch with comprehensive logging."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_desc = f"Epoch {epoch+1}/{total_epochs} [Training]"
        with tqdm(train_loader, desc=progress_desc, unit="batch") as pbar:
            for batch_idx, (code1, code2, labels) in enumerate(pbar):
                # Move data to device
                code1 = code1.to(self.device)
                code2 = code2.to(self.device)
                labels = labels.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                similarity = self.model(code1, code2)
                loss = self.criterion(similarity, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Track statistics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                avg_loss = total_loss / num_batches
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        epoch_loss = total_loss / num_batches
        self.train_losses.append(epoch_loss)
        return epoch_loss

    def validate_epoch(self, val_loader, epoch, total_epochs):
        """Validate the model with detailed metrics."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_desc = f"Epoch {epoch+1}/{total_epochs} [Validation]"
        with torch.no_grad():
            with tqdm(val_loader, desc=progress_desc, unit="batch") as pbar:
                for code1, code2, labels in pbar:
                    # Move data to device
                    code1 = code1.to(self.device)
                    code2 = code2.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    similarity = self.model(code1, code2)
                    loss = self.criterion(similarity, labels)
                    
                    # Predictions
                    predictions = (similarity >= 0.5).float()
                    
                    # Collect for metrics
                    total_loss += loss.item()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Update progress bar
                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_labels, all_predictions)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Store metrics
        self.val_losses.append(avg_loss)
        self.val_f1_scores.append(f1)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, f1, accuracy

    def save_checkpoint(self, epoch, f1_score, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'f1_score': f1_score,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'val_accuracies': self.val_accuracies,
        }
        
        if is_best:
            self.best_model_path = f"siamese_epoch_{epoch+1}_f1_{f1_score:.4f}.pth"
            torch.save(checkpoint, self.best_model_path)
            print(f"   Saved best model: {self.best_model_path}")
        
        # Always save latest
        torch.save(checkpoint, "siamese_latest.pth")

    def train(self, train_loader, val_loader, epochs, patience=5):
        """
        Complete training loop with validation and early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Early stopping patience: {patience}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss = self.train_epoch(train_loader, epoch, epochs)
            
            # Validation phase
            val_loss, val_f1, val_accuracy = self.validate_epoch(val_loader, epoch, epochs)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"   Training Loss: {train_loss:.4f}")
            print(f"   Validation Loss: {val_loss:.4f}")
            print(f"   Validation F1: {val_f1:.4f}")
            print(f"   Validation Accuracy: {val_accuracy:.4f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for best model
            is_best = val_f1 > self.best_f1
            if is_best:
                self.best_f1 = val_f1
                self.patience_counter = 0
                print(f"   New best F1 score!")
            else:
                self.patience_counter += 1
                print(f"   No improvement (patience: {self.patience_counter}/{patience})")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_f1, is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"Total epochs: {len(self.train_losses)}")
        print(f"Best validation F1: {self.best_f1:.4f}")
        print(f"Final training loss: {self.train_losses[-1]:.4f}")
        print(f"Final validation loss: {self.val_losses[-1]:.4f}")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Best model saved as: {self.best_model_path}")
        
        # Print training summary
        self.print_training_summary()

    def print_training_summary(self):
        """Print comprehensive training summary."""
        print(f"\nTraining metrics summary:")
        print(f"   Training losses: {[f'{loss:.4f}' for loss in self.train_losses]}")
        print(f"   Validation F1 scores: {[f'{f1:.4f}' for f1 in self.val_f1_scores]}")
        print(f"   Validation accuracies: {[f'{acc:.4f}' for acc in self.val_accuracies]}")

def main():
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data with enhanced pipeline
    print("Loading BigCloneBench dataset...")
    train_dataset, val_dataset, test_dataset, tokenizer = load_bigclonebench_data_robust(
        subset_size=35000,
        max_seq_length=512,
        validation_split=0.15
    )
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Data loaded successfully:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Initialize model with enhanced architecture
    model = ImprovedSiameseNetwork(
        vocab_size=tokenizer.vocab_size,
        embed_size=256,
        hidden_size=256
    ).to(device)
    
    # Enhanced loss function and optimizer
    criterion = AdvancedContrastiveLoss(margin=1.0, temperature=0.1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    print(f"Model initialized:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer and start training
    trainer = AdvancedTrainer(model, criterion, optimizer, scheduler, device)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        patience=3
    )

if __name__ == "__main__":
    main()
