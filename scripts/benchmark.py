#!/usr/bin/env python3
"""
Benchmark script for comparing different zero-shot classification models.

This script provides comprehensive benchmarking capabilities for evaluating
and comparing different models and configurations.
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.classifiers import CLIPZeroShotClassifier, EnsembleZeroShotClassifier
from data.dataset import ZeroShotDataModule, create_sample_dataset
from eval.metrics import ZeroShotEvaluator, Leaderboard, benchmark_models
from utils.config import set_seed, get_device

logger = logging.getLogger(__name__)


def run_comprehensive_benchmark(data_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Run comprehensive benchmark of all available models.
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing benchmark results
    """
    logger.info("Starting comprehensive benchmark...")
    
    # Setup data
    data_module = ZeroShotDataModule(
        data_dir=data_dir,
        batch_size=16,
        num_workers=4
    )
    data_module.setup()
    
    # Get labels
    labels = data_module.train_dataset.dataset.get_unique_labels()
    logger.info(f"Found {len(labels)} unique labels: {labels}")
    
    # Define models to benchmark
    models_to_test = {
        "CLIP-ViT-B/32": "openai/clip-vit-base-patch32",
        "CLIP-ViT-B/16": "openai/clip-vit-base-patch16",
        "CLIP-ViT-L/14": "openai/clip-vit-large-patch14"
    }
    
    # Load models
    models = {}
    device = str(get_device())
    
    for model_name, model_path in models_to_test.items():
        try:
            logger.info(f"Loading {model_name}...")
            model = CLIPZeroShotClassifier(model_name=model_path, device=device)
            models[model_name] = model
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
    
    # Create ensemble if we have multiple models
    if len(models) >= 2:
        try:
            ensemble_model = EnsembleZeroShotClassifier(
                models=list(models.values()),
                weights=None
            )
            models["Ensemble"] = ensemble_model
            logger.info("Created ensemble model")
        except Exception as e:
            logger.warning(f"Failed to create ensemble: {e}")
    
    # Run benchmark
    leaderboard = Leaderboard(save_path=output_dir / "leaderboard.json")
    results = benchmark_models(models, data_module.test_dataloader(), labels, leaderboard)
    
    # Generate visualizations
    generate_benchmark_plots(results, output_dir)
    
    # Save detailed results
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def generate_benchmark_plots(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate visualization plots for benchmark results.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save plots
    """
    logger.info("Generating benchmark visualizations...")
    
    # Prepare data for plotting
    model_names = list(results.keys())
    metrics = ['accuracy', 'top_3_accuracy', 'top_5_accuracy']
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        
        bars = axes[i].bar(model_names, values)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create efficiency plot
    if any('efficiency' in results[model] for model in model_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Inference time
        times = []
        throughputs = []
        
        for model in model_names:
            if 'efficiency' in results[model]:
                eff = results[model]['efficiency']
                times.append(eff['avg_time_per_sample'])
                throughputs.append(eff['samples_per_second'])
            else:
                times.append(0)
                throughputs.append(0)
        
        ax1.bar(model_names, times)
        ax1.set_title('Average Inference Time per Sample')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(model_names, throughputs)
        ax2.set_title('Throughput (Samples per Second)')
        ax2.set_ylabel('Samples/sec')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "efficiency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def run_ablation_study(data_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Run ablation study on different configurations.
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing ablation results
    """
    logger.info("Starting ablation study...")
    
    # Setup data
    data_module = ZeroShotDataModule(data_dir=data_dir, batch_size=16)
    data_module.setup()
    
    labels = data_module.train_dataset.dataset.get_unique_labels()
    
    # Test different label templates
    label_templates = {
        "Simple": ["dog", "cat", "car", "bird", "flower"],
        "Descriptive": ["a photo of a dog", "a photo of a cat", "a photo of a car", 
                       "a photo of a bird", "a photo of a flower"],
        "Detailed": ["a photo of a golden retriever dog", "a photo of a persian cat",
                    "a photo of a red sports car", "a photo of a blue jay bird",
                    "a photo of a red rose flower"]
    }
    
    model = CLIPZeroShotClassifier(device=str(get_device()))
    evaluator = ZeroShotEvaluator(model, device=str(get_device()))
    
    ablation_results = {}
    
    for template_name, template_labels in label_templates.items():
        logger.info(f"Testing {template_name} template...")
        
        # Filter labels to match template
        filtered_labels = [label for label in template_labels if any(gt in label.lower() for gt in labels)]
        
        if filtered_labels:
            metrics = evaluator.evaluate(data_module.test_dataloader(), filtered_labels)
            ablation_results[template_name] = metrics
    
    # Save ablation results
    with open(output_dir / "ablation_results.json", 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    return ablation_results


def generate_report(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate a comprehensive benchmark report.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save report
    """
    logger.info("Generating benchmark report...")
    
    report_lines = []
    report_lines.append("# Zero-shot Image Classification Benchmark Report")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Summary table
    report_lines.append("## Model Performance Summary")
    report_lines.append("")
    report_lines.append("| Model | Accuracy | Top-3 Acc | Top-5 Acc | Inference Time |")
    report_lines.append("|-------|----------|-----------|-----------|----------------|")
    
    for model_name, metrics in results.items():
        accuracy = metrics.get('accuracy', 0)
        top3_acc = metrics.get('top_3_accuracy', 0)
        top5_acc = metrics.get('top_5_accuracy', 0)
        
        if 'efficiency' in metrics:
            inference_time = metrics['efficiency']['avg_time_per_sample']
        else:
            inference_time = 0
        
        report_lines.append(f"| {model_name} | {accuracy:.4f} | {top3_acc:.4f} | {top5_acc:.4f} | {inference_time:.4f}s |")
    
    report_lines.append("")
    
    # Detailed analysis
    report_lines.append("## Detailed Analysis")
    report_lines.append("")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1].get('accuracy', 0))
    report_lines.append(f"**Best Performing Model:** {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    report_lines.append("")
    
    # Efficiency analysis
    if any('efficiency' in metrics for metrics in results.values()):
        fastest_model = min(
            [(name, metrics) for name, metrics in results.items() if 'efficiency' in metrics],
            key=lambda x: x[1]['efficiency']['avg_time_per_sample']
        )
        report_lines.append(f"**Fastest Model:** {fastest_model[0]} ({fastest_model[1]['efficiency']['avg_time_per_sample']:.4f}s per sample)")
        report_lines.append("")
    
    # Save report
    with open(output_dir / "benchmark_report.md", 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to {output_dir / 'benchmark_report.md'}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Zero-shot classification benchmark")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--output-dir", type=str, default="results/benchmark",
                       help="Output directory for results")
    parser.add_argument("--mode", type=str, default="comprehensive",
                       choices=["comprehensive", "ablation", "quick"],
                       help="Benchmark mode")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample dataset if it doesn't exist")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Set seed for reproducibility
    set_seed()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample dataset if needed
    if args.create_sample or not data_dir.exists():
        logger.info("Creating sample dataset...")
        create_sample_dataset(data_dir, num_classes=5, samples_per_class=20)
    
    # Run benchmark
    if args.mode == "comprehensive":
        results = run_comprehensive_benchmark(data_dir, output_dir)
        generate_report(results, output_dir)
    elif args.mode == "ablation":
        results = run_ablation_study(data_dir, output_dir)
    else:  # quick
        logger.info("Running quick benchmark...")
        # Simplified benchmark for quick testing
        results = run_comprehensive_benchmark(data_dir, output_dir)
    
    logger.info("Benchmark completed successfully!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
