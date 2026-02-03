"""
Strong Lensing Classification Model using ResNet18

Classifies strong lensing images into three categories:
1. No substructure
2. CDM subhalos
3. Superfluid dark matter vortices
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class StrongLensingDataset(Dataset):
    """Dataset class for strong lensing images."""
    
    def __init__(self, 
                 data_dir: str = "subhalo_dataset",
                 split: str = "train",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 seed: int = 42,
                 transform=None):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        data_dir : str
            Directory containing images and labels
        split : str
            One of 'train', 'val', or 'test'
        train_ratio : float
            Fraction of data for training
        val_ratio : float
            Fraction of data for validation
        seed : int
            Random seed for reproducibility
        transform : callable
            Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Category mapping
        self.categories = {
            'no_substructure': 0,
            'cdm_subhalos': 1,
            'sfdm_vortices': 2
        }
        
        self.category_names = ['No Substructure', 'CDM Subhalos', 'SFDM Vortices']
        
        # Load all image paths and labels
        self.samples = []
        
        for category, label in self.categories.items():
            image_files = sorted((self.data_dir / "images").glob(f"{category}_*.npy"))
            
            for img_path in image_files:
                self.samples.append({
                    'image_path': img_path,
                    'label': label,
                    'category': category
                })
        
        # Split data
        np.random.seed(seed)
        n_samples = len(self.samples)
        indices = np.random.permutation(n_samples)
        
        n_train = int(train_ratio * n_samples)
        n_val = int(val_ratio * n_samples)
        
        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        elif split == 'test':
            self.indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        print(f"{split.upper()} set: {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]
        
        # Load image
        image = np.load(sample['image_path'])
        
        # Normalize image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Convert to 3-channel (RGB) for ResNet
        image = np.stack([image, image, image], axis=0)
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = sample['label']
        
        return image, label


class LensingClassifier(nn.Module):
    """ResNet18-based classifier for strong lensing images."""
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        """
        Initialize the classifier.
        
        Parameters
        ----------
        num_classes : int
            Number of output classes (default: 3)
        pretrained : bool
            Whether to use pretrained ImageNet weights
        """
        super(LensingClassifier, self).__init__()
        
        # Load ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer to accept single-channel input if needed
        # (though we convert to 3-channel, keeping this for flexibility)
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        """Forward pass."""
        return self.resnet(x)


class Trainer:
    """Training manager for the lensing classifier."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: str = 'cuda',
                 save_dir: str = 'models'):
        """
        Initialize the trainer.
        
        Parameters
        ----------
        model : nn.Module
            The neural network model
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        criterion : nn.Module
            Loss function
        optimizer : optim.Optimizer
            Optimizer
        device : str
            Device to train on ('cuda' or 'cpu')
        save_dir : str
            Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs: int = 50):
        """
        Train the model for multiple epochs.
        
        Parameters
        ----------
        num_epochs : int
            Number of epochs to train
        """
        best_val_acc = 0.0
        
        print(f"\nTraining on {self.device}")
        print("-" * 70)
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(
                    epoch,
                    val_acc,
                    filename='best_model.pth'
                )
                print(f"  → Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print("-" * 70)
        print(f"Training complete! Best Val Acc: {best_val_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, epoch: int, val_acc: float, filename: str = 'checkpoint.pth'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str = 'best_model.pth'):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
              f"with Val Acc: {checkpoint['val_acc']:.2f}%")


class Evaluator:
    """Evaluation and visualization utilities."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        model : nn.Module
            Trained model
        device : str
            Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.category_names = ['No Substructure', 'CDM Subhalos', 'SFDM Vortices']
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test set.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test data loader
            
        Returns
        -------
        results : dict
            Dictionary containing predictions, true labels, and metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = 100. * (all_predictions == all_labels).sum() / len(all_labels)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=self.category_names,
            output_dict=True
        )
        
        results = {
            'predictions': all_predictions,
            'labels': all_labels,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.category_names,
            yticklabels=self.category_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history: Dict, save_path: str = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_predictions(self, 
                            test_loader: DataLoader,
                            num_samples: int = 9,
                            save_path: str = None):
        """Visualize predictions on test samples."""
        self.model.eval()
        
        # Get a batch
        images, labels = next(iter(test_loader))
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = outputs.max(1)
        
        # Move to CPU for plotting
        images = images.cpu()
        predicted = predicted.cpu()
        
        # Plot
        n_cols = 3
        n_rows = (num_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(images))):
            ax = axes[i]
            
            # Display image (take first channel)
            img = images[i, 0].numpy()
            ax.imshow(img, cmap='hot', origin='lower')
            
            # Title with prediction
            true_label = self.category_names[labels[i]]
            pred_label = self.category_names[predicted[i]]
            color = 'green' if labels[i] == predicted[i] else 'red'
            
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
            ax.axis('off')
        
        # Hide extra subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main training and evaluation pipeline."""
    # Configuration
    DATA_DIR = "subhalo_dataset"
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("Strong Lensing Classification with ResNet18")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    # Create datasets
    train_dataset = StrongLensingDataset(
        data_dir=DATA_DIR,
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )
    
    val_dataset = StrongLensingDataset(
        data_dir=DATA_DIR,
        split='val',
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )
    
    test_dataset = StrongLensingDataset(
        data_dir=DATA_DIR,
        split='test',
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    # Create model
    model = LensingClassifier(num_classes=3, pretrained=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        save_dir='models'
    )
    
    # Train
    history = trainer.train(num_epochs=NUM_EPOCHS)
    
    # Load best model for evaluation
    trainer.load_checkpoint('best_model.pth')
    
    # Evaluate
    evaluator = Evaluator(model=trainer.model, device=DEVICE)
    
    # Plot training history
    evaluator.plot_training_history(history, save_path='training_history.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluator.evaluate(test_loader)
    
    print(f"\nTest Accuracy: {results['accuracy']:.2f}%")
    print("\nClassification Report:")
    for category, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"\n{category}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1-score']:.3f}")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        save_path='confusion_matrix.png'
    )
    
    # Visualize predictions
    evaluator.visualize_predictions(
        test_loader,
        num_samples=9,
        save_path='predictions.png'
    )
    
    print("\n" + "=" * 70)
    print("Training and evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
