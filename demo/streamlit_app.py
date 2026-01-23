"""
Interactive demo for zero-shot image classification using Streamlit.

This module provides a web-based interface for testing zero-shot image
classification models with real-time predictions and visualizations.
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.classifiers import CLIPZeroShotClassifier, EnsembleZeroShotClassifier
from utils.config import get_device, set_seed
from eval.metrics import ZeroShotEvaluator

logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Zero-shot Image Classification",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name: str = "openai/clip-vit-base-patch32"):
    """Load and cache the CLIP model.
    
    Args:
        model_name: Name of the CLIP model to load
        
    Returns:
        Loaded CLIP model
    """
    device = get_device()
    model = CLIPZeroShotClassifier(model_name=model_name, device=device)
    return model


def create_sample_labels() -> List[str]:
    """Create sample label templates.
    
    Returns:
        List of sample label templates
    """
    return [
        "a photo of a dog",
        "a photo of a cat", 
        "a photo of a car",
        "a photo of a bird",
        "a photo of a flower",
        "a photo of a tree",
        "a photo of a house",
        "a photo of a person",
        "a photo of a bicycle",
        "a photo of an airplane",
        "a photo of food",
        "a photo of a book",
        "a photo of a phone",
        "a photo of a computer",
        "a photo of a chair"
    ]


def plot_prediction_confidence(probabilities: torch.Tensor, labels: List[str]) -> None:
    """Plot prediction confidence scores.
    
    Args:
        probabilities: Probability distribution over labels
        labels: List of labels
    """
    probs = probabilities.cpu().numpy()
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(sorted_labels)), sorted_probs)
    
    # Color bars by confidence
    colors = plt.cm.viridis(sorted_probs)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Prediction Confidence Scores')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', va='center', ha='left')
    
    plt.tight_layout()
    st.pyplot(fig)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ Zero-shot Image Classification</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases zero-shot image classification using CLIP (Contrastive Language-Image Pretraining).
    Upload an image and provide candidate labels to see how well the model can classify images
    without having seen examples of those specific classes during training.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_options = {
        "CLIP ViT-B/32": "openai/clip-vit-base-patch32",
        "CLIP ViT-B/16": "openai/clip-vit-base-patch16",
        "CLIP ViT-L/14": "openai/clip-vit-large-patch14"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0
    )
    
    model_name = model_options[selected_model]
    
    # Load model
    with st.spinner(f"Loading {selected_model}..."):
        model = load_model(model_name)
    
    st.sidebar.success(f"✅ {selected_model} loaded successfully!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to classify"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"Image size: {image.size[0]}x{image.size[1]} pixels")
    
    with col2:
        st.header("🏷️ Candidate Labels")
        
        # Label input options
        label_input_mode = st.radio(
            "Label Input Mode",
            ["Custom Labels", "Sample Labels", "Preset Categories"],
            index=1
        )
        
        labels = []
        
        if label_input_mode == "Custom Labels":
            st.text_area(
                "Enter labels (one per line)",
                value="a photo of a dog\na photo of a cat\na photo of a car",
                height=150,
                help="Enter candidate labels, one per line. Use descriptive phrases like 'a photo of a dog' for better results."
            )
            custom_labels = st.text_area("Custom labels").strip()
            if custom_labels:
                labels = [line.strip() for line in custom_labels.split('\n') if line.strip()]
        
        elif label_input_mode == "Sample Labels":
            sample_labels = create_sample_labels()
            selected_samples = st.multiselect(
                "Select sample labels",
                options=sample_labels,
                default=sample_labels[:5]
            )
            labels = selected_samples
        
        else:  # Preset Categories
            categories = {
                "Animals": ["a photo of a dog", "a photo of a cat", "a photo of a bird", "a photo of a fish"],
                "Vehicles": ["a photo of a car", "a photo of a truck", "a photo of a bicycle", "a photo of a motorcycle"],
                "Objects": ["a photo of a chair", "a photo of a table", "a photo of a book", "a photo of a phone"],
                "Nature": ["a photo of a tree", "a photo of a flower", "a photo of a mountain", "a photo of a lake"]
            }
            
            selected_category = st.selectbox("Select category", list(categories.keys()))
            labels = categories[selected_category]
        
        # Display selected labels
        if labels:
            st.write("**Selected Labels:**")
            for i, label in enumerate(labels, 1):
                st.write(f"{i}. {label}")
    
    # Classification
    if uploaded_file is not None and labels:
        st.header("🔍 Classification Results")
        
        # Perform classification
        with st.spinner("Classifying image..."):
            try:
                prediction, probabilities = model.classify(image, labels, return_probs=True)
                
                # Display results
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.metric("Predicted Label", prediction)
                
                with col2:
                    confidence = torch.max(probabilities).item()
                    st.metric("Confidence", f"{confidence:.3f}")
                
                with col3:
                    top3_conf = torch.topk(probabilities, 3)[0].sum().item()
                    st.metric("Top-3 Confidence", f"{top3_conf:.3f}")
                
                # Detailed results
                st.subheader("📊 Detailed Results")
                
                # Top predictions
                top_k = min(5, len(labels))
                top_indices = torch.topk(probabilities, top_k).indices
                
                st.write("**Top Predictions:**")
                for i, idx in enumerate(top_indices):
                    label = labels[idx.item()]
                    prob = probabilities[idx].item()
                    st.write(f"{i+1}. **{label}** - {prob:.3f}")
                
                # Confidence visualization
                st.subheader("📈 Confidence Visualization")
                plot_prediction_confidence(probabilities, labels)
                
                # Model information
                with st.expander("ℹ️ Model Information"):
                    st.write(f"**Model:** {selected_model}")
                    st.write(f"**Device:** {model.device}")
                    st.write(f"**Number of Labels:** {len(labels)}")
                    st.write(f"**Image Size:** {image.size}")
                
            except Exception as e:
                st.error(f"Error during classification: {str(e)}")
                logger.error(f"Classification error: {e}")
    
    # Batch processing section
    st.header("📁 Batch Processing")
    
    with st.expander("Process Multiple Images"):
        st.write("Upload multiple images for batch classification:")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files and labels:
            if st.button("Process All Images"):
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {i+1}/{len(uploaded_files)}")
                    
                    try:
                        image = Image.open(uploaded_file).convert('RGB')
                        prediction, probabilities = model.classify(image, labels, return_probs=True)
                        
                        results.append({
                            'filename': uploaded_file.name,
                            'prediction': prediction,
                            'confidence': torch.max(probabilities).item(),
                            'probabilities': probabilities
                        })
                        
                    except Exception as e:
                        results.append({
                            'filename': uploaded_file.name,
                            'prediction': 'Error',
                            'confidence': 0.0,
                            'error': str(e)
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display batch results
                st.subheader("📋 Batch Results")
                
                for result in results:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{result['filename']}**")
                    
                    with col2:
                        st.write(f"Prediction: {result['prediction']}")
                    
                    with col3:
                        st.write(f"Confidence: {result['confidence']:.3f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About Zero-shot Classification:**
    
    Zero-shot learning allows models to classify images into categories they haven't seen during training.
    CLIP achieves this by learning a joint embedding space for images and text, enabling it to understand
    semantic relationships between visual and textual representations.
    
    **Tips for better results:**
    - Use descriptive labels (e.g., "a photo of a dog" instead of just "dog")
    - Include diverse candidate labels
    - Ensure good image quality and resolution
    - Consider the semantic similarity between your labels
    """)


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed()
    
    # Run the app
    main()
