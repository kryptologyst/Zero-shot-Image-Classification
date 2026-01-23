#!/usr/bin/env python3
"""
Utility script for downloading and preparing datasets for zero-shot classification.

This script helps download common datasets and convert them to the format
expected by the zero-shot classification system.
"""

import argparse
import logging
import sys
from pathlib import Path
import requests
import zipfile
import tarfile
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.dataset import create_sample_dataset
from utils.config import set_seed

logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
    """
    logger.info(f"Downloading {url} to {output_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Download completed: {output_path}")


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract an archive file.
    
    Args:
        archive_path: Path to the archive
        extract_dir: Directory to extract to
    """
    logger.info(f"Extracting {archive_path} to {extract_dir}")
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    logger.info(f"Extraction completed: {extract_dir}")


def download_cifar10(data_dir: Path) -> None:
    """Download and prepare CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to save the dataset
    """
    logger.info("Preparing CIFAR-10 dataset...")
    
    # CIFAR-10 classes
    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # For this demo, we'll create a sample dataset
    # In practice, you would download the actual CIFAR-10 dataset
    create_sample_dataset(data_dir / "cifar10", num_classes=len(classes), samples_per_class=50)
    
    logger.info(f"CIFAR-10 dataset prepared in {data_dir / 'cifar10'}")


def download_imagenet_sample(data_dir: Path) -> None:
    """Download a sample of ImageNet classes.
    
    Args:
        data_dir: Directory to save the dataset
    """
    logger.info("Preparing ImageNet sample dataset...")
    
    # Common ImageNet classes
    classes = [
        'dog', 'cat', 'bird', 'fish', 'horse',
        'car', 'truck', 'airplane', 'ship', 'bicycle',
        'chair', 'table', 'book', 'phone', 'computer'
    ]
    
    create_sample_dataset(data_dir / "imagenet_sample", num_classes=len(classes), samples_per_class=30)
    
    logger.info(f"ImageNet sample prepared in {data_dir / 'imagenet_sample'}")


def download_custom_dataset(data_dir: Path, dataset_name: str, 
                          num_classes: int = 10, samples_per_class: int = 20) -> None:
    """Download a custom sample dataset.
    
    Args:
        data_dir: Directory to save the dataset
        dataset_name: Name of the dataset
        num_classes: Number of classes
        samples_per_class: Samples per class
    """
    logger.info(f"Preparing custom dataset: {dataset_name}")
    
    create_sample_dataset(data_dir / dataset_name, num_classes=num_classes, samples_per_class=samples_per_class)
    
    logger.info(f"Custom dataset prepared in {data_dir / dataset_name}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Dataset preparation utility")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--dataset", type=str, default="sample",
                       choices=["cifar10", "imagenet_sample", "custom", "sample"],
                       help="Dataset to prepare")
    parser.add_argument("--num-classes", type=int, default=5,
                       help="Number of classes (for custom dataset)")
    parser.add_argument("--samples-per-class", type=int, default=20,
                       help="Samples per class (for custom dataset)")
    parser.add_argument("--dataset-name", type=str, default="custom",
                       help="Name for custom dataset")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Set seed for reproducibility
    set_seed()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare dataset
    if args.dataset == "cifar10":
        download_cifar10(data_dir)
    elif args.dataset == "imagenet_sample":
        download_imagenet_sample(data_dir)
    elif args.dataset == "custom":
        download_custom_dataset(
            data_dir, 
            args.dataset_name,
            args.num_classes,
            args.samples_per_class
        )
    else:  # sample
        download_custom_dataset(
            data_dir,
            "sample",
            args.num_classes,
            args.samples_per_class
        )
    
    logger.info("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
