"""
Zero-shot Image Classification Models

This module contains implementations of various models for zero-shot image classification,
including CLIP-based models and few-shot learning baselines.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CLIPZeroShotClassifier(nn.Module):
    """Zero-shot image classifier using CLIP model.
    
    This class wraps the CLIP model to provide a clean interface for zero-shot
    image classification tasks.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        """Initialize the CLIP zero-shot classifier.
        
        Args:
            model_name: Name of the CLIP model to use
            device: Device to run the model on
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Move to device
        self.model = self.model.to(device)
        
        logger.info(f"Loaded CLIP model: {model_name} on {device}")
    
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode text into embeddings.
        
        Args:
            text: Text or list of texts to encode
            
        Returns:
            Text embeddings tensor
        """
        if isinstance(text, str):
            text = [text]
        
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features
    
    def encode_image(self, image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Encode image(s) into embeddings.
        
        Args:
            image: PIL Image or list of PIL Images
            
        Returns:
            Image embeddings tensor
        """
        if isinstance(image, Image.Image):
            image = [image]
        
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def classify(self, image: Image.Image, labels: List[str], 
                 return_probs: bool = True) -> Union[str, Tuple[str, torch.Tensor]]:
        """Classify an image using zero-shot learning.
        
        Args:
            image: PIL Image to classify
            labels: List of candidate labels
            return_probs: Whether to return probability distribution
            
        Returns:
            Predicted label, optionally with probabilities
        """
        # Encode image and text
        image_features = self.encode_image(image)
        text_features = self.encode_text(labels)
        
        # Compute similarities
        similarities = torch.matmul(image_features, text_features.T)
        probs = F.softmax(similarities * 100, dim=1)  # Temperature scaling
        
        # Get prediction
        pred_idx = torch.argmax(probs, dim=1).item()
        predicted_label = labels[pred_idx]
        
        if return_probs:
            return predicted_label, probs[0]
        else:
            return predicted_label
    
    def forward(self, images: List[Image.Image], labels: List[str]) -> Dict[str, torch.Tensor]:
        """Forward pass for batch processing.
        
        Args:
            images: List of PIL Images
            labels: List of candidate labels
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        # Encode images and text
        image_features = self.encode_image(images)
        text_features = self.encode_text(labels)
        
        # Compute similarities
        similarities = torch.matmul(image_features, text_features.T)
        probs = F.softmax(similarities * 100, dim=1)
        
        # Get predictions
        pred_indices = torch.argmax(probs, dim=1)
        predictions = [labels[idx.item()] for idx in pred_indices]
        
        return {
            'predictions': predictions,
            'probabilities': probs,
            'similarities': similarities
        }


class FewShotBaseline(nn.Module):
    """Few-shot learning baseline using metric learning.
    
    This class implements a simple few-shot learning baseline that can be used
    as a comparison to zero-shot methods.
    """
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 1000):
        """Initialize the few-shot baseline.
        
        Args:
            feature_dim: Dimension of input features
            num_classes: Number of classes for classification
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Simple linear classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        logger.info(f"Initialized few-shot baseline with {num_classes} classes")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Input features tensor
            
        Returns:
            Classification logits
        """
        return self.classifier(features)
    
    def compute_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes for few-shot learning.
        
        Args:
            features: Feature vectors
            labels: Corresponding labels
            
        Returns:
            Class prototypes
        """
        unique_labels = torch.unique(labels)
        prototypes = []
        
        for label in unique_labels:
            mask = labels == label
            prototype = features[mask].mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def classify_by_prototypes(self, query_features: torch.Tensor, 
                              support_features: torch.Tensor, 
                              support_labels: torch.Tensor) -> torch.Tensor:
        """Classify query features using prototype-based classification.
        
        Args:
            query_features: Features to classify
            support_features: Support set features
            support_labels: Support set labels
            
        Returns:
            Predicted labels
        """
        # Compute prototypes
        prototypes = self.compute_prototypes(support_features, support_labels)
        unique_labels = torch.unique(support_labels)
        
        # Compute distances to prototypes
        distances = torch.cdist(query_features, prototypes)
        
        # Get predictions
        pred_indices = torch.argmin(distances, dim=1)
        predictions = unique_labels[pred_indices]
        
        return predictions


class EnsembleZeroShotClassifier(nn.Module):
    """Ensemble of multiple zero-shot classifiers.
    
    This class combines multiple zero-shot classifiers to improve performance
    through ensemble methods.
    """
    
    def __init__(self, models: List[CLIPZeroShotClassifier], 
                 weights: Optional[List[float]] = None):
        """Initialize ensemble classifier.
        
        Args:
            models: List of zero-shot classifiers
            weights: Optional weights for each model
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
        
        logger.info(f"Initialized ensemble with {len(models)} models")
    
    def classify(self, image: Image.Image, labels: List[str]) -> Tuple[str, torch.Tensor]:
        """Classify using ensemble of models.
        
        Args:
            image: PIL Image to classify
            labels: List of candidate labels
            
        Returns:
            Predicted label and ensemble probabilities
        """
        all_probs = []
        
        # Get predictions from each model
        for model in self.models:
            _, probs = model.classify(image, labels, return_probs=True)
            all_probs.append(probs)
        
        # Weighted average of probabilities
        ensemble_probs = torch.zeros_like(all_probs[0])
        for probs, weight in zip(all_probs, self.weights):
            ensemble_probs += weight * probs
        
        # Get final prediction
        pred_idx = torch.argmax(ensemble_probs).item()
        predicted_label = labels[pred_idx]
        
        return predicted_label, ensemble_probs


def create_model(model_type: str = "clip", **kwargs) -> nn.Module:
    """Factory function to create models.
    
    Args:
        model_type: Type of model to create
        **kwargs: Additional arguments for model creation
        
    Returns:
        Initialized model
    """
    if model_type == "clip":
        return CLIPZeroShotClassifier(**kwargs)
    elif model_type == "few_shot":
        return FewShotBaseline(**kwargs)
    elif model_type == "ensemble":
        models = kwargs.get('models', [])
        weights = kwargs.get('weights', None)
        return EnsembleZeroShotClassifier(models, weights)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    import sys
    sys.path.append(".")
    
    from utils.config import get_device
    
    device = get_device()
    
    # Test CLIP model
    print("Testing CLIP Zero-shot Classifier...")
    model = CLIPZeroShotClassifier(device=device)
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    labels = ["a photo of a dog", "a photo of a cat", "a photo of a car"]
    
    prediction, probs = model.classify(dummy_image, labels)
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {probs}")
    
    print("Model test completed successfully!")
