"""
Zero-shot Image Classification Project

A modern, reproducible implementation of zero-shot image classification using
CLIP and other vision-language models for research and education purposes.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


class Config:
    """Configuration class for the zero-shot classification project."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.
        
        Args:
            config_dict: Optional dictionary to override default config
        """
        # Default configuration
        self.seed = 42
        self.device = get_device()
        
        # Model configuration
        self.model_name = "openai/clip-vit-base-patch32"
        self.batch_size = 32
        self.max_length = 77
        
        # Data configuration
        self.data_dir = Path("data")
        self.image_size = 224
        self.num_workers = 4
        
        # Evaluation configuration
        self.top_k = 5
        self.confidence_threshold = 0.1
        
        # Paths
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
        self.assets_dir = Path("assets")
        
        # Override with provided config
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    if config_path and Path(config_path).exists():
        # TODO: Implement YAML/JSON config loading
        logger.info(f"Loading config from {config_path}")
        return Config()
    else:
        logger.info("Using default configuration")
        return Config()


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed()
    
    # Load configuration
    config = load_config()
    
    # Print configuration
    logger.info("Configuration loaded:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
