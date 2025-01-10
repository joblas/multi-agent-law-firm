import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
import os

class LegalDataset(Dataset):
    """Custom Dataset for legal data"""
    
    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model: BaseEstimator, batch_size: int = 32):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.model = model
        self.batch_size = batch_size
        self.training_history = []
        self.best_metrics = {}
        
        # Create directory for saving models if it doesn't exist
        os.makedirs('models/saved', exist_ok=True)
        
    def train(self, 
              train_data: Dict[str, pd.DataFrame],
              epochs: int = 10,
              learning_rate: float = 0.001,
              early_stopping_patience: int = 3) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_data: Dictionary containing train, validation, and test DataFrames
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary containing training metrics and history
        """
        try:
            # Prepare data loaders
            train_loader = self._prepare_data_loader(train_data['train'])
            val_loader = self._prepare_data_loader(train_data['validation'])
            
            # Initialize optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = self._train_epoch(train_loader, optimizer)
                
                # Validation phase
                self.model.eval()
                val_loss, val_metrics = self._validate_epoch(val_loader)
                
                # Log metrics
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val Accuracy: {val_metrics['accuracy']:.4f}"
                )
                
                # Save training history
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **val_metrics
                })
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_metrics = val_metrics
                    self._save_model('best_model.pt')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break
                    
            return {
                'training_history': self.training_history,
                'best_metrics': self.best_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error in train: {str(e)}")
            raise
            
    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch"""
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_features)
            loss = torch.nn.functional.cross_entropy(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
        
    def _validate_epoch(self, val_loader: DataLoader) -> tuple:
        """Validate for one epoch"""
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                # Forward pass
                outputs = self.model(batch_features)
                loss = torch.nn.functional.cross_entropy(outputs, batch_labels)
                
                # Store predictions and labels
                predictions = outputs.argmax(dim=1)
                all_predictions.extend(predictions.numpy())
                all_labels.extend(batch_labels.numpy())
                
                total_loss += loss.item()
                
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions)
        
        return total_loss / len(val_loader), metrics
        
    def _calculate_metrics(self, true_labels: List[int], predictions: List[int]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def _prepare_data_loader(self, df: pd.DataFrame) -> DataLoader:
        """Prepare DataLoader from DataFrame"""
        features = df.drop('label', axis=1).values
        labels = df['label'].values
        
        dataset = LegalDataset(features, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_data: Test DataFrame
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            self.model.eval()
            test_loader = self._prepare_data_loader(test_data)
            
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    outputs = self.model(batch_features)
                    predictions = outputs.argmax(dim=1)
                    
                    all_predictions.extend(predictions.numpy())
                    all_labels.extend(batch_labels.numpy())
                    
            metrics = self._calculate_metrics(all_labels, all_predictions)
            
            self.logger.info("Test Metrics:")
            for metric_name, value in metrics.items():
                self.logger.info(f"{metric_name}: {value:.4f}")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in evaluate: {str(e)}")
            raise
            
    def _save_model(self, filename: str):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'best_metrics': self.best_metrics,
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat()
            }
            
            save_path = os.path.join('models/saved', filename)
            torch.save(checkpoint, save_path)
            self.logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            
    def load_model(self, filename: str):
        """Load model checkpoint"""
        try:
            load_path = os.path.join('models/saved', filename)
            checkpoint = torch.load(load_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_metrics = checkpoint['best_metrics']
            self.training_history = checkpoint['training_history']
            
            self.logger.info(f"Model loaded from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    def save_training_history(self, filename: str = 'training_history.json'):
        """Save training history to JSON"""
        try:
            save_path = os.path.join('models/saved', filename)
            
            history = {
                'training_history': self.training_history,
                'best_metrics': self.best_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(save_path, 'w') as f:
                json.dump(history, f, indent=4)
                
            self.logger.info(f"Training history saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving training history: {str(e)}")
