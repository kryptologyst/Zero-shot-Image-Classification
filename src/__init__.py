"""
Zero-shot Image Classification Package

A modern, reproducible implementation of zero-shot image classification
using CLIP and other vision-language models for research and education.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

from .models.classifiers import (
    CLIPZeroShotClassifier,
    FewShotBaseline,
    EnsembleZeroShotClassifier,
    create_model
)

from .data.dataset import (
    ZeroShotDataset,
    ZeroShotDataModule,
    create_sample_dataset
)

from .eval.metrics import (
    ZeroShotEvaluator,
    Leaderboard,
    benchmark_models
)

from .utils.config import (
    set_seed,
    get_device,
    Config,
    load_config
)

__all__ = [
    # Models
    "CLIPZeroShotClassifier",
    "FewShotBaseline", 
    "EnsembleZeroShotClassifier",
    "create_model",
    
    # Data
    "ZeroShotDataset",
    "ZeroShotDataModule",
    "create_sample_dataset",
    
    # Evaluation
    "ZeroShotEvaluator",
    "Leaderboard",
    "benchmark_models",
    
    # Utils
    "set_seed",
    "get_device",
    "Config",
    "load_config",
]
