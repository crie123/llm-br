import os
import sys
from pathlib import Path

# Add parent directory to Python path to find local modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import wasserstein_distance
import joblib
import torch
from train_bitgrid_decoder import MLPDecoder
from bitgrid import BitGridSensor, image_to_bitgrid, phase_to_bitgrid

def _ensure_flat_bits(bits, g=16):
    a = np.asarray(bits).ravel().astype(np.float32)
    total = g * g
    if a.size >= total:
        return a[:total]
    out = np.zeros(total, dtype=np.float32)
    out[:a.size] = a
    return out


def load_trained_decoder(decoder_pt='bitgrid_decoder.pt', meta_joblib='bitgrid_decoder_meta.joblib', device='cpu'):
    if not os.path.exists(decoder_pt) or not os.path.exists(meta_joblib):
        return None, None, None
    try:
        prep = joblib.load(meta_joblib)
        scaler = prep.get('scaler', None)
        pca = prep.get('pca', None)
        data = torch.load(decoder_pt, map_location='cpu')
        model_state = data.get('model_state', data)
        input_dim = data.get('input_dim', None)
        output_dim = data.get('output_dim', None)
        hidden = tuple(data.get('hidden', (512, 256)))
        if input_dim is None or output_dim is None:
            return None, scaler, pca
        model = MLPDecoder(input_dim, hidden_dims=hidden, output_dim=output_dim).to(device)
        model.load_state_dict(model_state)
        model.eval()
        return model, scaler, pca
    except Exception:
        return None, None, None


def plot_phase_reconstruction(img_path, sensor=None, decoder_model=None, decoder_scaler=None, decoder_pca=None):
    """Plot original image, bit grid, phase reconstruction, and metrics"""
    if sensor is None:
        sensor = BitGridSensor(dim=128, grid=16)
        
    # Load and preprocess image
    img = np.array(Image.open(img_path).convert('L')) / 255.0
    
    # Get features and reconstruction
    features = sensor.encode_image(img)
    
    # Try decoder prediction
    decoder_bits = None
    try:
        if decoder_model is not None:
            recon_flat = features['reconstruction'].ravel().astype(np.float32)
            phase_vec = np.asarray(features['phase']).ravel().astype(np.float32)
            moments = features['moments'].astype(np.float32)
            hist = features['hist_orig'].astype(np.float32)
            feat_vec = np.concatenate([recon_flat, phase_vec, moments, hist]).reshape(1, -1)
            if decoder_scaler is not None:
                feat_vec = decoder_scaler.transform(feat_vec)
            if decoder_pca is not None:
                feat_vec = decoder_pca.transform(feat_vec)
            with torch.no_grad():
                X_t = torch.tensor(feat_vec, dtype=torch.float32)
                logits = decoder_model(X_t)
                probs = torch.sigmoid(logits).cpu().numpy()
                decoder_bits = (probs >= 0.5).astype(np.float32).reshape(-1)
    except Exception:
        decoder_bits = None
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Phase Reconstruction Analysis', fontsize=14)
    
    # Original image
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title('Original Image')
    
    # Bit grid
    bits = np.asarray(features['bits']).reshape(sensor.grid, sensor.grid)
    axes[0,1].imshow(bits, cmap='gray')
    axes[0,1].set_title(f'Bit Grid ({sensor.grid}x{sensor.grid})')
    
    # Phase visualization
    phase = np.asarray(features['phase']).reshape(sensor.grid, sensor.grid)
    axes[0,2].imshow(phase, cmap='hsv')
    axes[0,2].set_title('Phase Pattern')
    
    # Reconstruction
    recon = features['reconstruction']
    axes[1,0].imshow(recon, cmap='gray')
    axes[1,0].set_title('Reconstructed Image')
    
    # Decoder mask or thresholded recon
    if decoder_bits is not None:
        mask = decoder_bits.reshape(sensor.grid, sensor.grid)
        axes[1,1].imshow(mask, cmap='gray')
        axes[1,1].set_title('Decoder Predicted Mask')
    else:
        threshold = float(np.percentile(recon, 60))
        mask = (recon >= threshold).astype(np.float32)
        axes[1,1].imshow(mask, cmap='gray')
        axes[1,1].set_title('Thresholded Recon Mask')
    
    # Reconstruction error
    if img.shape == recon.shape:
        error = np.abs(img - recon)
        axes[1,2].imshow(error, cmap='hot')
        axes[1,2].set_title('Reconstruction Error')
    else:
        axes[1,2].text(0.5, 0.5, 'Size mismatch', ha='center')
    
    # Metrics text overlay
    plt.tight_layout()
    return fig


def evaluate_image_pair(img1_path, img2_path, sensor=None, decoder_model=None, decoder_scaler=None, decoder_pca=None):
    """Compare two images using the BitGridSensor metrics"""
    if sensor is None:
        sensor = BitGridSensor(dim=128, grid=16)
        
    # Load images
    img1 = np.array(Image.open(img1_path).convert('L')) / 255.0
    img2 = np.array(Image.open(img2_path).convert('L')) / 255.0
    
    # Get features
    feat1 = sensor.encode_image(img1)
    feat2 = sensor.encode_image(img2)
    
    # Decoder predictions
    def _predict_decoder(features):
        try:
            recon_flat = features['reconstruction'].ravel().astype(np.float32)
            phase_vec = np.asarray(features['phase']).ravel().astype(np.float32)
            moments = features['moments'].astype(np.float32)
            hist = features['hist_orig'].astype(np.float32)
            feat_vec = np.concatenate([recon_flat, phase_vec, moments, hist]).reshape(1, -1)
            if decoder_scaler is not None:
                feat_vec = decoder_scaler.transform(feat_vec)
            if decoder_pca is not None:
                feat_vec = decoder_pca.transform(feat_vec)
            with torch.no_grad():
                X_t = torch.tensor(feat_vec, dtype=torch.float32)
                logits = decoder_model(X_t)
                probs = torch.sigmoid(logits).cpu().numpy()
                return (probs >= 0.5).astype(np.float32).reshape(-1)
        except Exception:
            return None

    dec1 = None
    dec2 = None
    if decoder_model is not None:
        dec1 = _predict_decoder(feat1)
        dec2 = _predict_decoder(feat2)

    # Compare patterns
    score, metrics = sensor.compare_patterns(feat1, feat2)
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Image Pair Comparison (Overall Score: {score:.3f})', fontsize=14)
    
    # First image row
    axes[0,0].imshow(img1, cmap='gray')
    axes[0,0].set_title('Image 1')
    
    axes[0,1].imshow(np.asarray(feat1['bits']).reshape(sensor.grid, sensor.grid), cmap='gray')
    axes[0,1].set_title('Bit Grid 1')
    
    axes[0,2].imshow(np.asarray(feat1['phase']).reshape(sensor.grid, sensor.grid), cmap='hsv')
    axes[0,2].set_title('Phase 1')
    
    axes[0,3].imshow(feat1['reconstruction'], cmap='gray')
    axes[0,3].set_title('Reconstruction 1')
    
    # Second image row
    axes[1,0].imshow(img2, cmap='gray')
    axes[1,0].set_title('Image 2')
    
    axes[1,1].imshow(np.asarray(feat2['bits']).reshape(sensor.grid, sensor.grid), cmap='gray')
    axes[1,1].set_title('Bit Grid 2')
    
    axes[1,2].imshow(np.asarray(feat2['phase']).reshape(sensor.grid, sensor.grid), cmap='hsv')
    axes[1,2].set_title('Phase 2')
    
    axes[1,3].imshow(feat2['reconstruction'], cmap='gray')
    axes[1,3].set_title('Reconstruction 2')

    # Decoder masks (or thresholded)
    if dec1 is not None:
        axes[0,3].imshow(dec1.reshape(sensor.grid, sensor.grid), cmap='gray')
        axes[0,3].set_title('Decoder Mask 1')
    if dec2 is not None:
        axes[1,3].imshow(dec2.reshape(sensor.grid, sensor.grid), cmap='gray')
        axes[1,3].set_title('Decoder Mask 2')

    # Add metrics text
    metrics_text = [
        f"Comparison Metrics:",
        f"Spectral Similarity: {metrics['spectral_similarity']:.3f}",
        f"Moment Distance: {metrics['moment_distance']:.3f}",
        f"Wasserstein (orig): {metrics['wasserstein_orig']:.3f}",
        f"Wasserstein (recon): {metrics['wasserstein_recon']:.3f}",
        f"IoU Score: {metrics['iou']:.3f}",
        f"Combined Score: {metrics['combined_score']:.3f}"
    ]
    
    # Add text box with metrics
    plt.figtext(0.02, 0.02, '\n'.join(metrics_text), 
                fontsize=10, family='monospace',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, metrics

if __name__ == "__main__":
    # Example usage
    sensor = BitGridSensor(dim=128, grid=16)
    # try load decoder
    decoder_model, decoder_scaler, decoder_pca = load_trained_decoder(decoder_pt='bitgrid_decoder.pt', meta_joblib='bitgrid_decoder_meta.joblib')
    
    # Single image analysis
    img_path = "cifar_bitgrid_example_3.png"  # Replace with actual path
    if os.path.exists(img_path):
        fig = plot_phase_reconstruction(img_path, sensor, decoder_model, decoder_scaler, decoder_pca)
        plt.show()
    
    # Image pair comparison
    img1_path = "cifar_bitgrid_example_0.png"  # Replace with actual path
    img2_path = "cifar_bitgrid_example_1.png"  # Replace with actual path
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        fig, metrics = evaluate_image_pair(img1_path, img2_path, sensor, decoder_model, decoder_scaler, decoder_pca)
        plt.show()