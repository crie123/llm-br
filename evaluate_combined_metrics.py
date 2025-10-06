import os
import numpy as np
import torch
from bitgrid import BitGridSensor
import matplotlib.pyplot as plt
import argparse
import logging

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

if __name__ == '__main__':
    # Example usage
    grid_size = 32
    p1 = np.random.rand(grid_size, grid_size) > 0.5
    p2 = np.random.rand(grid_size, grid_size) > 0.5
    
    score, metrics = evaluate_metrics(p1, p2)
    plot_comparison(p1, p2, metrics, "comparison.png")