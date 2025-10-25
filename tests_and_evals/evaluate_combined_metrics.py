import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import logging
from datasets import load_dataset
from collections import defaultdict
import sys

# Add parent directory to Python path to find local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bitgrid import BitGridSensor, image_to_bitgrid, phase_to_bitgrid

def evaluate_metrics(pattern1, pattern2, grid_size=32):
    """Evaluate similarity between two patterns using basic metrics."""
    sensor = BitGridSensor(grid_size)
    
    # Encode patterns using encode_image instead of encode_pattern
    feat1 = sensor.encode_image(pattern1)
    feat2 = sensor.encode_image(pattern2)
    
    # Compare and get metrics
    score, metrics = sensor.compare_patterns(feat1, feat2)
    
    return score, metrics

def plot_comparison(pattern1, pattern2, metrics, save_path=None):
    """Plot patterns and their comparison metrics."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(131)
    plt.imshow(pattern1, cmap='gray')
    plt.title('Pattern 1')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(pattern2, cmap='gray')
    plt.title('Pattern 2')
    plt.axis('off')
    
    plt.subplot(133)
    metrics_list = [
        f"Combined Score: {metrics['combined_score']:.3f}",
        f"Spectral Sim: {metrics['spectral_similarity']:.3f}",
        f"Moment Dist: {metrics['moment_distance']:.3f}",
        f"Wasserstein: {metrics['wasserstein_recon']:.3f}",
        f"IoU: {metrics['iou']:.3f}"
    ]
    plt.axis('off')
    plt.text(0.1, 0.5, '\n'.join(metrics_list), 
             fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_cifar_patterns(num_samples=1000, grid_size=32):
    """Evaluate pattern metrics on CIFAR-10 dataset and plot distributions."""
    try:
        ds = load_dataset('cifar10', split=f'train[:{num_samples}]')
    except Exception as e:
        print(f"Failed to load CIFAR-10: {e}")
        return None

    sensor = BitGridSensor(grid_size)
    metrics_dict = defaultdict(list)
    
    print(f"Evaluating {len(ds)} CIFAR-10 images...")
    for i in range(0, len(ds)-1, 2):
        try:
            # Get pairs of consecutive images
            img1 = np.array(ds[i]['img'].convert('L').resize((128, 128))) / 255.0
            img2 = np.array(ds[i+1]['img'].convert('L').resize((128, 128))) / 255.0
            
            score, metrics = evaluate_metrics(img1, img2, grid_size)
            
            # Collect all metrics
            for key, value in metrics.items():
                metrics_dict[key].append(value)
                
        except Exception as e:
            print(f"Error processing images {i},{i+1}: {e}")
            continue
    
    # Plot histograms of all metrics
    plt.figure(figsize=(15, 10))
    keys = ['combined_score', 'spectral_similarity', 'moment_distance', 'wasserstein_recon', 'iou']
    for idx, key in enumerate(keys, 1):
        plt.subplot(2, 3, idx)
        values = metrics_dict[key]
        if values:
            plt.hist(values, bins=30, alpha=0.7)
            plt.title(f'{key} Distribution')
            plt.xlabel('Value')
            plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('cifar_metrics_distribution.png')
    plt.close()
    
    # Print summary statistics
    print("\nMetrics Summary:")
    for key in keys:
        values = metrics_dict[key]
        if values:
            print(f"{key}:")
            print(f"  Mean: {np.mean(values):.3f}")
            print(f"  Std:  {np.std(values):.3f}")
    
    return metrics_dict

if __name__ == '__main__':
    # Example usage with CIFAR evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000, help='Number of CIFAR-10 samples to evaluate')
    parser.add_argument('--grid-size', type=int, default=32, help='Grid size for bit grid encoding')
    args = parser.parse_args()
    
    # Run CIFAR evaluation
    metrics = evaluate_cifar_patterns(args.samples, args.grid_size)
    
    if metrics is None:
        # Fallback to random pattern comparison
        grid_size = args.grid_size
        p1 = np.random.rand(grid_size, grid_size) > 0.5
        p2 = np.random.rand(grid_size, grid_size) > 0.5
        score, metrics = evaluate_metrics(p1, p2)
        plot_comparison(p1, p2, metrics, "comparison.png")