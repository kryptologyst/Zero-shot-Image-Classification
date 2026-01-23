"""
Unit tests for zero-shot image classification project.

This module contains comprehensive tests for all major components
of the zero-shot classification system.
"""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.classifiers import CLIPZeroShotClassifier, FewShotBaseline, EnsembleZeroShotClassifier
from data.dataset import ZeroShotDataset, ZeroShotDataModule, create_sample_dataset
from eval.metrics import ZeroShotEvaluator, Leaderboard
from utils.config import set_seed, get_device, Config


class TestConfig:
    """Test configuration utilities."""
    
    def test_set_seed(self):
        """Test seed setting functionality."""
        set_seed(42)
        
        # Test that seeds are set
        assert torch.initial_seed() is not None
        
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_config_creation(self):
        """Test configuration object creation."""
        config = Config()
        assert config.seed == 42
        assert isinstance(config.device, torch.device)
        
        # Test custom config
        custom_config = Config({'seed': 123, 'batch_size': 64})
        assert custom_config.seed == 123
        assert custom_config.batch_size == 64


class TestModels:
    """Test model implementations."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return Image.new('RGB', (224, 224), color='red')
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return ["a photo of a dog", "a photo of a cat", "a photo of a car"]
    
    def test_clip_classifier_initialization(self):
        """Test CLIP classifier initialization."""
        model = CLIPZeroShotClassifier(device="cpu")
        assert model.device == "cpu"
        assert model.model_name == "openai/clip-vit-base-patch32"
    
    def test_clip_classifier_classify(self, sample_image, sample_labels):
        """Test CLIP classifier classification."""
        model = CLIPZeroShotClassifier(device="cpu")
        
        prediction, probabilities = model.classify(sample_image, sample_labels)
        
        assert isinstance(prediction, str)
        assert prediction in sample_labels
        assert isinstance(probabilities, torch.Tensor)
        assert probabilities.shape[0] == len(sample_labels)
        assert torch.allclose(probabilities.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_clip_classifier_encode_text(self, sample_labels):
        """Test text encoding."""
        model = CLIPZeroShotClassifier(device="cpu")
        
        text_features = model.encode_text(sample_labels)
        
        assert isinstance(text_features, torch.Tensor)
        assert text_features.shape[0] == len(sample_labels)
        assert text_features.shape[1] > 0
    
    def test_clip_classifier_encode_image(self, sample_image):
        """Test image encoding."""
        model = CLIPZeroShotClassifier(device="cpu")
        
        image_features = model.encode_image(sample_image)
        
        assert isinstance(image_features, torch.Tensor)
        assert image_features.shape[0] == 1
        assert image_features.shape[1] > 0
    
    def test_few_shot_baseline(self):
        """Test few-shot baseline model."""
        model = FewShotBaseline(feature_dim=512, num_classes=10)
        
        # Test forward pass
        features = torch.randn(5, 512)
        logits = model(features)
        
        assert logits.shape == (5, 10)
    
    def test_few_shot_prototypes(self):
        """Test prototype computation."""
        model = FewShotBaseline(feature_dim=512, num_classes=10)
        
        features = torch.randn(10, 512)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        prototypes = model.compute_prototypes(features, labels)
        
        assert prototypes.shape[0] == 5  # 5 unique labels
        assert prototypes.shape[1] == 512
    
    def test_ensemble_classifier(self, sample_image, sample_labels):
        """Test ensemble classifier."""
        # Create multiple models
        models = [
            CLIPZeroShotClassifier(device="cpu"),
            CLIPZeroShotClassifier(device="cpu")
        ]
        
        ensemble = EnsembleZeroShotClassifier(models)
        
        prediction, probabilities = ensemble.classify(sample_image, sample_labels)
        
        assert isinstance(prediction, str)
        assert prediction in sample_labels
        assert isinstance(probabilities, torch.Tensor)


class TestDataModule:
    """Test data handling modules."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_create_sample_dataset(self, temp_data_dir):
        """Test sample dataset creation."""
        create_sample_dataset(temp_data_dir, num_classes=3, samples_per_class=5)
        
        # Check directory structure
        assert temp_data_dir.exists()
        
        class_dirs = list(temp_data_dir.iterdir())
        assert len(class_dirs) == 3
        
        for class_dir in class_dirs:
            assert class_dir.is_dir()
            images = list(class_dir.glob("*.jpg"))
            assert len(images) == 5
    
    def test_zero_shot_dataset(self, temp_data_dir):
        """Test zero-shot dataset."""
        # Create sample data
        create_sample_dataset(temp_data_dir, num_classes=2, samples_per_class=3)
        
        dataset = ZeroShotDataset(temp_data_dir)
        
        assert len(dataset) == 6  # 2 classes * 3 samples
        
        # Test getting a sample
        sample = dataset[0]
        assert 'image' in sample
        assert 'label' in sample
        assert 'class_id' in sample
        assert isinstance(sample['image'], torch.Tensor)
        assert sample['image'].shape[0] == 3  # RGB channels
    
    def test_zero_shot_data_module(self, temp_data_dir):
        """Test data module."""
        # Create sample data
        create_sample_dataset(temp_data_dir, num_classes=3, samples_per_class=10)
        
        data_module = ZeroShotDataModule(
            temp_data_dir,
            batch_size=4,
            num_workers=0  # Avoid multiprocessing issues in tests
        )
        data_module.setup()
        
        # Test data loaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        # Test a batch
        batch = next(iter(train_loader))
        assert 'image' in batch
        assert 'label' in batch
        assert batch['image'].shape[0] <= 4  # batch_size


class TestEvaluation:
    """Test evaluation metrics and utilities."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        return CLIPZeroShotClassifier(device="cpu")
    
    @pytest.fixture
    def sample_dataloader(self, temp_data_dir):
        """Create a sample dataloader for testing."""
        create_sample_dataset(temp_data_dir, num_classes=2, samples_per_class=5)
        
        data_module = ZeroShotDataModule(
            temp_data_dir,
            batch_size=2,
            num_workers=0
        )
        data_module.setup()
        
        return data_module.test_dataloader()
    
    def test_zero_shot_evaluator(self, sample_model, sample_dataloader):
        """Test zero-shot evaluator."""
        evaluator = ZeroShotEvaluator(sample_model, device="cpu")
        
        labels = ["dog", "cat"]
        metrics = evaluator.evaluate(sample_dataloader, labels)
        
        assert 'accuracy' in metrics
        assert 'top_3_accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confidence_stats' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_leaderboard(self):
        """Test leaderboard functionality."""
        leaderboard = Leaderboard()
        
        # Add entries
        metrics1 = {'accuracy': 0.85, 'top_3_accuracy': 0.92}
        metrics2 = {'accuracy': 0.87, 'top_3_accuracy': 0.94}
        
        leaderboard.add_entry("Model1", {}, metrics1, "Test run 1")
        leaderboard.add_entry("Model2", {}, metrics2, "Test run 2")
        
        # Test getting leaderboard
        df = leaderboard.get_leaderboard('accuracy')
        
        assert len(df) == 2
        assert df.iloc[0]['model_name'] == "Model2"  # Higher accuracy first
        assert df.iloc[0]['accuracy'] == 0.87


class TestIntegration:
    """Integration tests."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_classification(self, temp_data_dir):
        """Test end-to-end classification pipeline."""
        # Create sample data
        create_sample_dataset(temp_data_dir, num_classes=3, samples_per_class=5)
        
        # Setup data
        data_module = ZeroShotDataModule(
            temp_data_dir,
            batch_size=2,
            num_workers=0
        )
        data_module.setup()
        
        # Create model
        model = CLIPZeroShotClassifier(device="cpu")
        
        # Get labels
        labels = data_module.train_dataset.dataset.get_unique_labels()
        
        # Evaluate
        evaluator = ZeroShotEvaluator(model, device="cpu")
        metrics = evaluator.evaluate(data_module.test_dataloader(), labels)
        
        # Check that evaluation completed successfully
        assert 'accuracy' in metrics
        assert metrics['accuracy'] >= 0
        assert metrics['accuracy'] <= 1
    
    def test_model_consistency(self):
        """Test that models produce consistent results."""
        model = CLIPZeroShotClassifier(device="cpu")
        
        image = Image.new('RGB', (224, 224), color='blue')
        labels = ["a photo of a dog", "a photo of a cat"]
        
        # Run classification multiple times
        results = []
        for _ in range(3):
            prediction, probabilities = model.classify(image, labels)
            results.append((prediction, probabilities))
        
        # Check consistency
        predictions = [r[0] for r in results]
        probabilities = [r[1] for r in results]
        
        # All predictions should be the same
        assert all(p == predictions[0] for p in predictions)
        
        # Probabilities should be very close
        for prob in probabilities[1:]:
            assert torch.allclose(prob, probabilities[0], atol=1e-6)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
