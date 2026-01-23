#!/usr/bin/env python3
"""
Main training and evaluation script for zero-shot image classification.

This script provides a unified interface for training, evaluating, and benchmarking
zero-shot image classification models.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config import set_seed, get_device, Config
from models.classifiers import create_model, CLIPZeroShotClassifier, EnsembleZeroShotClassifier
from data.dataset import ZeroShotDataModule, create_sample_dataset
from eval.metrics import ZeroShotEvaluator, Leaderboard, benchmark_models

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def evaluate_model(model, data_module: ZeroShotDataModule, 
                  labels: list, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a model on the dataset.
    
    Args:
        model: Model to evaluate
        data_module: Data module containing test data
        labels: List of candidate labels
        config: Configuration dictionary
        
    Returns:
        Evaluation metrics
    """
    logger.info("Starting model evaluation...")
    
    # Create evaluator
    evaluator = ZeroShotEvaluator(model, device=config['model']['device'])
    
    # Get test data loader
    test_loader = data_module.test_dataloader()
    
    # Evaluate
    metrics = evaluator.evaluate(
        test_loader, 
        labels, 
        top_k_values=config['evaluation']['top_k_values']
    )
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    logger.info(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    
    if 'efficiency' in metrics:
        eff = metrics['efficiency']
        logger.info(f"Average inference time: {eff['avg_time_per_sample']:.4f}s")
        logger.info(f"Samples per second: {eff['samples_per_second']:.2f}")
    
    return metrics


def benchmark_multiple_models(config: Dict[str, Any]) -> None:
    """Benchmark multiple models and create leaderboard.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting multi-model benchmark...")
    
    # Create sample dataset if it doesn't exist
    data_dir = Path(config['data']['data_dir'])
    if not data_dir.exists():
        logger.info("Creating sample dataset...")
        create_sample_dataset(data_dir, num_classes=5, samples_per_class=20)
    
    # Setup data module
    data_module = ZeroShotDataModule(
        data_dir=data_dir,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split']
    )
    data_module.setup()
    
    # Get unique labels from dataset
    labels = data_module.train_dataset.dataset.get_unique_labels()
    logger.info(f"Found {len(labels)} unique labels: {labels}")
    
    # Create models to benchmark
    models = {}
    
    # CLIP models
    clip_models = [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch14"
    ]
    
    for model_name in clip_models:
        try:
            model = CLIPZeroShotClassifier(
                model_name=model_name,
                device=config['model']['device']
            )
            models[f"CLIP-{model_name.split('/')[-1]}"] = model
            logger.info(f"Loaded {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
    
    # Create ensemble model
    if len(models) >= 2:
        try:
            ensemble_model = EnsembleZeroShotClassifier(
                models=list(models.values()),
                weights=None  # Equal weights
            )
            models["Ensemble"] = ensemble_model
            logger.info("Created ensemble model")
        except Exception as e:
            logger.warning(f"Failed to create ensemble: {e}")
    
    # Benchmark models
    leaderboard = Leaderboard(save_path="results/leaderboard.json")
    results = benchmark_models(models, data_module.test_dataloader(), labels, leaderboard)
    
    # Display leaderboard
    logger.info("Benchmark Results:")
    df = leaderboard.get_leaderboard('accuracy')
    for _, row in df.iterrows():
        logger.info(f"{row['model_name']}: Accuracy={row['accuracy']:.4f}, "
                   f"Top-3={row['top_3_accuracy']:.4f}")
    
    # Save detailed results
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_dir / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_dir}")


def train_few_shot_baseline(config: Dict[str, Any]) -> None:
    """Train a few-shot learning baseline.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Training few-shot baseline...")
    
    # This is a placeholder for few-shot training
    # In practice, you would implement proper few-shot learning here
    logger.warning("Few-shot training not implemented yet")


def run_demo(config: Dict[str, Any]) -> None:
    """Run the interactive demo.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting interactive demo...")
    
    import subprocess
    import os
    
    demo_path = Path(__file__).parent / "demo" / "streamlit_app.py"
    
    if demo_path.exists():
        cmd = [
            "streamlit", "run", str(demo_path),
            "--server.port", str(config['demo']['streamlit']['port']),
            "--server.address", config['demo']['streamlit']['host']
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        logger.error(f"Demo file not found: {demo_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Zero-shot Image Classification")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="evaluate",
                       choices=["evaluate", "benchmark", "train", "demo"],
                       help="Mode to run")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                       help="Model to use for evaluation")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model != "openai/clip-vit-base-patch32":
        config['model']['name'] = args.model
    if args.data_dir != "data":
        config['data']['data_dir'] = args.data_dir
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Set device
    if config['model']['device'] == "auto":
        config['model']['device'] = str(get_device())
    
    logger.info(f"Running in {args.mode} mode")
    logger.info(f"Using device: {config['model']['device']}")
    
    # Create necessary directories
    for path_key in ['checkpoint_dir', 'log_dir', 'assets_dir', 'results_dir']:
        Path(config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
    
    # Run selected mode
    if args.mode == "evaluate":
        # Single model evaluation
        logger.info("Running single model evaluation...")
        
        # Create sample dataset if needed
        data_dir = Path(config['data']['data_dir'])
        if not data_dir.exists():
            create_sample_dataset(data_dir, num_classes=5, samples_per_class=20)
        
        # Setup data
        data_module = ZeroShotDataModule(
            data_dir=data_dir,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            image_size=config['data']['image_size']
        )
        data_module.setup()
        
        # Create model
        model = CLIPZeroShotClassifier(
            model_name=config['model']['name'],
            device=config['model']['device']
        )
        
        # Get labels
        labels = data_module.train_dataset.dataset.get_unique_labels()
        
        # Evaluate
        metrics = evaluate_model(model, data_module, labels, config)
        
    elif args.mode == "benchmark":
        # Multi-model benchmark
        benchmark_multiple_models(config)
        
    elif args.mode == "train":
        # Train few-shot baseline
        train_few_shot_baseline(config)
        
    elif args.mode == "demo":
        # Run interactive demo
        run_demo(config)
    
    logger.info("Execution completed successfully!")


if __name__ == "__main__":
    main()
