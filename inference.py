"""
CNN-VAE Steel Microstructure Analysis - Inference Script
========================================================
Use this script to load a trained model and predict properties 
from microstructure images.

Usage:
    python inference.py --image path/to/microstructure.png
"""

import torch
import argparse
import os
from cnn_vae_steel import (
    Config, CNNVAE, load_trained_model, 
    predict_from_image, print_predictions
)

def main():
    parser = argparse.ArgumentParser(
        description='Predict steel properties from microstructure image'
    )
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Path to microstructure image'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='cnn_vae_steel.pth',
        help='Path to trained model'
    )
    parser.add_argument(
        '--scaler', 
        type=str, 
        default='target_scaler.pkl',
        help='Path to scaler file'
    )
    parser.add_argument(
        '--show-image', 
        action='store_true',
        help='Display the input image'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("CNN-VAE Steel Microstructure Analysis - Inference")
    print("="*80)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model, scaler = load_trained_model(args.model, args.scaler)
    
    # Adjust image path if necessary
    image_path = args.image
    if not os.path.exists(image_path):
        image_path = os.path.join('data', args.image)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {args.image} or {image_path}")
    
    # Predict
    print(f"\nAnalyzing microstructure: {image_path}")
    results = predict_from_image(
        image_path, 
        model, 
        scaler, 
        show_image=args.show_image
    )
    
    # Print results
    print_predictions(results)
    
    # Save results to JSON
    import json
    output_file = image_path.replace('.png', '_predictions.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    main()