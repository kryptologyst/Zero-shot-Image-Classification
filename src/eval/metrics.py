"""
Evaluation metrics and utilities for zero-shot image classification.

This module provides comprehensive evaluation metrics, leaderboards, and
analysis tools for zero-shot image classification tasks.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import top_k_accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class ZeroShotEvaluator:
    """Evaluator for zero-shot image classification tasks.
    
    This class provides comprehensive evaluation metrics and analysis tools
    for zero-shot classification models.
    """
    
    def __init__(self, model, device: str = "cpu"):
        """Initialize the evaluator.
        
        Args:
            model: Zero-shot classification model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.results = {}
        
        logger.info("Initialized zero-shot evaluator")
    
    def evaluate(self, dataloader, labels: List[str], 
                top_k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """Evaluate the model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            labels: List of candidate labels
            top_k_values: List of k values for top-k accuracy
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_ground_truth = []
        all_confidences = []
        
        total_time = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image']
                ground_truth_labels = batch['label']
                
                # Move to device
                images = images.to(self.device)
                
                # Time inference
                start_time = time.time()
                
                # Get predictions
                if hasattr(self.model, 'forward'):
                    outputs = self.model.forward(images, labels)
                    predictions = outputs['predictions']
                    probabilities = outputs['probabilities']
                else:
                    # Handle single image classification
                    predictions = []
                    probabilities = []
                    for img in images:
                        pred, probs = self.model.classify(img, labels, return_probs=True)
                        predictions.append(pred)
                        probabilities.append(probs)
                    probabilities = torch.stack(probabilities)
                
                inference_time = time.time() - start_time
                total_time += inference_time
                num_samples += len(images)
                
                # Store results
                all_predictions.extend(predictions)
                all_probabilities.append(probabilities.cpu())
                all_ground_truth.extend(ground_truth_labels)
                
                # Compute confidences
                max_probs = torch.max(probabilities, dim=1)[0]
                all_confidences.extend(max_probs.cpu().tolist())
        
        # Convert to tensors
        all_probabilities = torch.cat(all_probabilities, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(
            all_predictions, all_ground_truth, all_probabilities, 
            labels, top_k_values
        )
        
        # Add efficiency metrics
        metrics['efficiency'] = {
            'total_time': total_time,
            'num_samples': num_samples,
            'avg_time_per_sample': total_time / num_samples,
            'samples_per_second': num_samples / total_time
        }
        
        self.results = metrics
        return metrics
    
    def _compute_metrics(self, predictions: List[str], ground_truth: List[str],
                        probabilities: torch.Tensor, labels: List[str],
                        top_k_values: List[int]) -> Dict[str, Any]:
        """Compute evaluation metrics.
        
        Args:
            predictions: List of predicted labels
            ground_truth: List of ground truth labels
            probabilities: Probability distributions
            labels: List of candidate labels
            top_k_values: List of k values for top-k accuracy
            
        Returns:
            Dictionary containing computed metrics
        """
        metrics = {}
        
        # Basic accuracy
        accuracy = accuracy_score(ground_truth, predictions)
        metrics['accuracy'] = accuracy
        
        # Top-k accuracy
        label_to_idx = {label: i for i, label in enumerate(labels)}
        y_true_indices = [label_to_idx[label] for label in ground_truth]
        
        for k in top_k_values:
            top_k_acc = top_k_accuracy_score(
                y_true_indices, probabilities.numpy(), k=k, labels=list(range(len(labels)))
            )
            metrics[f'top_{k}_accuracy'] = top_k_acc
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            ground_truth, predictions, average=None, zero_division=0
        )
        
        metrics['per_class'] = {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist(),
            'support': support_per_class.tolist(),
            'labels': labels
        }
        
        # Confidence statistics
        confidences = torch.max(probabilities, dim=1)[0]
        metrics['confidence_stats'] = {
            'mean': float(torch.mean(confidences)),
            'std': float(torch.std(confidences)),
            'min': float(torch.min(confidences)),
            'max': float(torch.max(confidences))
        }
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions, labels=labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix.
        
        Args:
            save_path: Optional path to save the plot
        """
        if 'confusion_matrix' not in self.results:
            logger.warning("No confusion matrix available. Run evaluation first.")
            return
        
        cm = np.array(self.results['confusion_matrix'])
        labels = self.results['per_class']['labels']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    def plot_confidence_distribution(self, save_path: Optional[str] = None) -> None:
        """Plot confidence score distribution.
        
        Args:
            save_path: Optional path to save the plot
        """
        if 'confidence_stats' not in self.results:
            logger.warning("No confidence stats available. Run evaluation first.")
            return
        
        # This would need access to individual confidences
        # For now, just show the statistics
        stats = self.results['confidence_stats']
        
        plt.figure(figsize=(8, 6))
        plt.bar(['Mean', 'Std', 'Min', 'Max'], 
               [stats['mean'], stats['std'], stats['min'], stats['max']])
        plt.title('Confidence Score Statistics')
        plt.ylabel('Confidence Score')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report.
        
        Returns:
            Formatted evaluation report
        """
        if not self.results:
            return "No evaluation results available. Run evaluation first."
        
        report = []
        report.append("=" * 50)
        report.append("ZERO-SHOT CLASSIFICATION EVALUATION REPORT")
        report.append("=" * 50)
        
        # Overall metrics
        report.append(f"\nOverall Accuracy: {self.results['accuracy']:.4f}")
        report.append(f"Precision: {self.results['precision']:.4f}")
        report.append(f"Recall: {self.results['recall']:.4f}")
        report.append(f"F1-Score: {self.results['f1_score']:.4f}")
        
        # Top-k accuracy
        report.append("\nTop-K Accuracy:")
        for key, value in self.results.items():
            if key.startswith('top_') and key.endswith('_accuracy'):
                k = key.replace('top_', '').replace('_accuracy', '')
                report.append(f"  Top-{k}: {value:.4f}")
        
        # Confidence statistics
        if 'confidence_stats' in self.results:
            stats = self.results['confidence_stats']
            report.append(f"\nConfidence Statistics:")
            report.append(f"  Mean: {stats['mean']:.4f}")
            report.append(f"  Std: {stats['std']:.4f}")
            report.append(f"  Min: {stats['min']:.4f}")
            report.append(f"  Max: {stats['max']:.4f}")
        
        # Efficiency metrics
        if 'efficiency' in self.results:
            eff = self.results['efficiency']
            report.append(f"\nEfficiency Metrics:")
            report.append(f"  Total Time: {eff['total_time']:.2f}s")
            report.append(f"  Avg Time per Sample: {eff['avg_time_per_sample']:.4f}s")
            report.append(f"  Samples per Second: {eff['samples_per_second']:.2f}")
        
        return "\n".join(report)


class Leaderboard:
    """Leaderboard for comparing different models and configurations.
    
    This class maintains a leaderboard of model performance across different
    metrics and configurations.
    """
    
    def __init__(self, save_path: Optional[str] = None):
        """Initialize the leaderboard.
        
        Args:
            save_path: Optional path to save leaderboard data
        """
        self.save_path = save_path
        self.entries = []
        
        logger.info("Initialized leaderboard")
    
    def add_entry(self, model_name: str, config: Dict[str, Any], 
                  metrics: Dict[str, Any], notes: str = "") -> None:
        """Add an entry to the leaderboard.
        
        Args:
            model_name: Name of the model
            config: Model configuration
            metrics: Evaluation metrics
            notes: Optional notes about the run
        """
        entry = {
            'timestamp': time.time(),
            'model_name': model_name,
            'config': config,
            'metrics': metrics,
            'notes': notes
        }
        
        self.entries.append(entry)
        
        # Save if path is provided
        if self.save_path:
            self.save()
        
        logger.info(f"Added entry for {model_name}")
    
    def get_leaderboard(self, metric: str = 'accuracy', 
                       ascending: bool = False) -> pd.DataFrame:
        """Get leaderboard sorted by a specific metric.
        
        Args:
            metric: Metric to sort by
            ascending: Whether to sort in ascending order
            
        Returns:
            DataFrame with leaderboard entries
        """
        if not self.entries:
            return pd.DataFrame()
        
        # Extract data for DataFrame
        data = []
        for entry in self.entries:
            row = {
                'timestamp': entry['timestamp'],
                'model_name': entry['model_name'],
                'notes': entry['notes']
            }
            
            # Add metrics
            if 'metrics' in entry:
                for key, value in entry['metrics'].items():
                    if isinstance(value, (int, float)):
                        row[key] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=ascending)
        
        return df
    
    def save(self) -> None:
        """Save leaderboard to file."""
        if self.save_path:
            with open(self.save_path, 'w') as f:
                json.dump(self.entries, f, indent=2)
            logger.info(f"Saved leaderboard to {self.save_path}")
    
    def load(self) -> None:
        """Load leaderboard from file."""
        if self.save_path and Path(self.save_path).exists():
            with open(self.save_path, 'r') as f:
                self.entries = json.load(f)
            logger.info(f"Loaded leaderboard from {self.save_path}")
    
    def plot_metric_comparison(self, metric: str = 'accuracy', 
                             save_path: Optional[str] = None) -> None:
        """Plot comparison of models by metric.
        
        Args:
            metric: Metric to compare
            save_path: Optional path to save the plot
        """
        df = self.get_leaderboard(metric)
        
        if df.empty:
            logger.warning("No data available for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        plt.bar(df['model_name'], df[metric])
        plt.title(f'Model Comparison: {metric.title()}')
        plt.xlabel('Model')
        plt.ylabel(metric.title())
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()


def benchmark_models(models: Dict[str, Any], dataloader, 
                    labels: List[str], leaderboard: Optional[Leaderboard] = None) -> Dict[str, Dict[str, Any]]:
    """Benchmark multiple models and add to leaderboard.
    
    Args:
        models: Dictionary of model name -> model instance
        dataloader: Data loader for evaluation
        labels: List of candidate labels
        leaderboard: Optional leaderboard to add results to
        
    Returns:
        Dictionary of results for each model
    """
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        evaluator = ZeroShotEvaluator(model)
        metrics = evaluator.evaluate(dataloader, labels)
        
        results[model_name] = metrics
        
        # Add to leaderboard if provided
        if leaderboard:
            leaderboard.add_entry(
                model_name=model_name,
                config={'model_type': type(model).__name__},
                metrics=metrics,
                notes=f"Benchmark run for {model_name}"
            )
    
    return results


if __name__ == "__main__":
    # Test the evaluation module
    import sys
    sys.path.append(".")
    
    from models.classifiers import CLIPZeroShotClassifier
    from data.dataset import create_sample_dataset, ZeroShotDataModule
    from utils.config import get_device
    import tempfile
    
    device = get_device()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample dataset
        create_sample_dataset(temp_dir, num_classes=3, samples_per_class=10)
        
        # Setup data
        data_module = ZeroShotDataModule(temp_dir, batch_size=4)
        data_module.setup()
        
        # Create model
        model = CLIPZeroShotClassifier(device=device)
        
        # Test evaluation
        labels = ["dog", "cat", "car"]
        evaluator = ZeroShotEvaluator(model, device)
        
        val_loader = data_module.val_dataloader()
        metrics = evaluator.evaluate(val_loader, labels)
        
        print("Evaluation completed successfully!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
        
        # Test leaderboard
        leaderboard = Leaderboard()
        leaderboard.add_entry("CLIP-Base", {}, metrics, "Test run")
        
        df = leaderboard.get_leaderboard()
        print(f"Leaderboard entries: {len(df)}")
        
        print("Evaluation module test completed successfully!")
