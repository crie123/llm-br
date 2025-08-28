import numpy as np
import math
import torch
from PIL import Image
import os

# Lazy decoder cache
_BITGRID_DECODER = None
_BITGRID_DECODER_TRIED = False
_BITGRID_DECODER_PATH = os.path.join(os.path.dirname(__file__), 'bitgrid_decoder.joblib')


def _cleanup_debug_files():
    """Remove legacy debug artifacts produced earlier to avoid workspace clutter."""
    try:
        for fname in os.listdir('.'):
            if fname.startswith('bitgrid_debug_') and (fname.endswith('.npz') or fname.endswith('.png')):
                try:
                    os.remove(fname)
                except Exception:
                    pass
        # also remove phase list file if present
        try:
            if os.path.exists('bitgrid_debug_phases.npy'):
                os.remove('bitgrid_debug_phases.npy')
        except Exception:
            pass
    except Exception:
        pass


def _get_decoder():
    """Lazily load the trained decoder (joblib). Returns decoder or None.
    Caches result and avoids repeated load attempts.
    """
    global _BITGRID_DECODER, _BITGRID_DECODER_TRIED
    if _BITGRID_DECODER is not None:
        return _BITGRID_DECODER
    if _BITGRID_DECODER_TRIED:
        return None
    _BITGRID_DECODER_TRIED = True
    if not os.path.exists(_BITGRID_DECODER_PATH):
        return None
    try:
        # joblib is available in project libs; import lazily
        import joblib
        data = joblib.load(_BITGRID_DECODER_PATH)
        # stored object might be dict {'clf': clf, 'grid': g}
        if isinstance(data, dict) and 'clf' in data:
            clf = data['clf']
        else:
            clf = data
        _BITGRID_DECODER = clf
        # cleanup old debug files now that decoder exists
        _cleanup_debug_files()
        return _BITGRID_DECODER
    except Exception:
        _BITGRID_DECODER = None
        return None


def image_to_bitgrid(img: np.ndarray, g: int = 16, threshold: float = 0.5) -> np.ndarray:
    """Convert image [H,W] or [H,W,C] to a GxG bit grid via average pooling and thresholding.
    Returns flattened array of length G*G with values {0,1}.

    Note: g is the number of grid cells per side (so output length == g*g).
    """
    if isinstance(img, (str,)):
        img = np.asarray(Image.open(img).convert('L'), dtype=np.float32) / 255.0
    if hasattr(img, 'dtype') and img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    H, W = img.shape
    # pad so H and W are divisible by g (g = number of cells per side)
    pad_h = (g - (H % g)) % g
    pad_w = (g - (W % g)) % g
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
        H, W = img.shape
    # block size in pixels for each grid cell
    block_h = H // g
    block_w = W // g
    # reshape into (g, block_h, g, block_w) and average over pixel blocks
    pooled = img.reshape((g, block_h, g, block_w)).mean(axis=(1, 3))
    pooled = (pooled - pooled.min()) / (np.ptp(pooled) + 1e-6)
    bits = (pooled >= threshold).astype(np.float32)
    return bits.flatten()


def bitgrid_to_phase(bits: np.ndarray, dim: int) -> torch.Tensor:
    g2 = int(math.sqrt(len(bits)))
    grid = bits.reshape(g2, g2)
    spec = np.fft.fft2(grid)
    phase = np.angle(spec).flatten()
    phase = np.interp(phase, (-math.pi, math.pi), (-1, 1))
    t = torch.tensor(phase, dtype=torch.float32)
    if t.numel() < dim:
        t = torch.cat([t, torch.zeros(dim - t.numel())], dim=0)
    else:
        t = t[:dim]
    return t.unsqueeze(0)  # (1, dim)


def phase_reconstruction(phase, g: int = 16, sigma: float = 0.8):
    """Return (recon_norm, recon_smoothed, threshold) for given phase vector.
    recon_norm is clipped to [0,1] and has shape (g,g).
    """
    if isinstance(phase, torch.Tensor):
        arr = phase.detach().cpu().numpy()
    else:
        arr = np.asarray(phase)
    phi = np.clip(arr, -1, 1) * math.pi
    L = g * g
    if phi.size < L:
        phi = np.pad(phi, (0, L - phi.size))
    else:
        phi = phi[:L]
    spec = np.exp(1j * phi).reshape(g, g)
    recon = np.fft.ifft2(spec).real

    # smoothing
    try:
        from scipy.ndimage import gaussian_filter
        recon_smoothed = gaussian_filter(recon, sigma=sigma)
    except Exception:
        try:
            from scipy.signal import convolve2d
            k = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
            k = k / k.sum()
            recon_smoothed = convolve2d(recon, k, mode='same', boundary='symm')
        except Exception:
            recon_smoothed = recon

    # robust percentile normalization
    p25, p75 = np.percentile(recon_smoothed, [25, 75])
    if (p75 - p25) <= 1e-9:
        recon_norm = (recon_smoothed - recon_smoothed.min()) / (np.ptp(recon_smoothed) + 1e-9)
    else:
        recon_norm = (recon_smoothed - p25) / (p75 - p25 + 1e-9)
        recon_norm = np.clip(recon_norm, 0.0, 1.0)

    threshold = float(np.median(recon_norm))
    return recon_norm, recon_smoothed, threshold


def phase_to_bitgrid(phase, g: int = 16) -> np.ndarray:
    """Convert phase -> bitgrid using phase_reconstruction and adaptive threshold.
    If a trained decoder (bitgrid_decoder.joblib) is present it will be used to
    predict bits from the normalized reconstruction; otherwise fallback to
    adaptive thresholding.
    """
    recon_norm, recon_smoothed, threshold = phase_reconstruction(phase, g=g)

    # Try to use trained decoder if available
    clf = _get_decoder()
    if clf is not None:
        try:
            feat = recon_norm.flatten().astype(np.float32).reshape(1, -1)
            pred = clf.predict(feat)
            if hasattr(pred, 'shape'):
                bits = pred.reshape(-1).astype(np.float32)
            else:
                bits = np.array(pred, dtype=np.float32).flatten()
            return bits
        except Exception:
            # on any decoder failure, fallback to thresholding below
            pass

    bits = (recon_norm >= threshold).astype(np.float32)

    # post-processing configurable (default: none)
    postproc = os.environ.get('BITGRID_POSTPROC', 'none').lower()
    try:
        if postproc in ('median', 'median_filter'):
            from scipy.ndimage import median_filter
            bits = median_filter(bits, size=(3, 3)).astype(np.float32)
        elif postproc in ('morph', 'closing', 'opening'):
            from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
            struct = generate_binary_structure(2, 1)
            if postproc == 'opening':
                bits = binary_opening(bits.astype(bool), structure=struct).astype(np.float32)
            else:
                bits = binary_closing(bits.astype(bool), structure=struct).astype(np.float32)

        remove_small = int(os.environ.get('BITGRID_MIN_AREA', '2'))
        if remove_small > 0:
            from scipy.ndimage import label
            labeled, ncomp = label(bits.astype(bool))
            if ncomp > 0:
                counts = np.bincount(labeled.flatten())
                mask = np.zeros_like(bits, dtype=bool)
                for comp_id in range(1, counts.size):
                    if counts[comp_id] >= remove_small:
                        mask |= (labeled == comp_id)
                bits = mask.astype(np.float32)
    except Exception:
        pass

    return bits.flatten()


class BitGridSensor:
    """Simple wrapper exposing image->phase encoding for training integration.
    If no images are available for a dataset entry, sensor.encode_none() returns zeros.
    """
    def __init__(self, dim: int = 128, grid: int = 16, threshold: float = 0.5):
        self.dim = dim
        self.grid = grid
        self.threshold = threshold

    def encode_image(self, img) -> torch.Tensor:
        try:
            if isinstance(img, Image.Image):
                arr = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
            elif isinstance(img, str):
                arr = np.asarray(Image.open(img).convert('L'), dtype=np.float32) / 255.0
            else:
                arr = np.asarray(img, dtype=np.float32)
            bits = image_to_bitgrid(arr, g=self.grid, threshold=self.threshold)
            return bitgrid_to_phase(bits, dim=self.dim).squeeze(0)
        except Exception:
            return torch.zeros(self.dim, dtype=torch.float32)

    def encode_none(self) -> torch.Tensor:
        return torch.zeros(self.dim, dtype=torch.float32)
