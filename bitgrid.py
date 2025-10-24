import numpy as np
import math
import torch
from PIL import Image
import os
from scipy.special import rel_entr
from scipy.stats import entropy, wasserstein_distance
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy import ndimage

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
    if (_BITGRID_DECODER_TRIED):
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
    """Convert image to a GxG bit grid robustly by resizing to (g,g) and thresholding.
    Returns flattened array of length G*G with values {0,1}.
    """
    # Accept PIL Image or numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    img = img.copy()

    # Normalize to [0,1]
    if img.dtype == np.uint8 or img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    # Convert color to grayscale by averaging channels
    if img.ndim == 3 and img.shape[2] > 1:
        img_gray = img.mean(axis=2)
    else:
        img_gray = img

    # Resize to (g,g) using PIL to avoid pooling reshape issues
    try:
        from PIL import Image
        pil = Image.fromarray((img_gray * 255).astype(np.uint8))
        pil_small = pil.resize((g, g), resample=Image.BILINEAR)
        small = np.array(pil_small).astype(np.float32) / 255.0
    except Exception:
        # Fallback: simple block pooling
        h, w = img_gray.shape
        cell_h = max(1, h // g)
        cell_w = max(1, w // g)
        small = np.zeros((g, g), dtype=np.float32)
        for i in range(g):
            for j in range(g):
                y0 = min(i * cell_h, h)
                x0 = min(j * cell_w, w)
                y1 = min((i + 1) * cell_h, h)
                x1 = min((j + 1) * cell_w, w)
                if y1 > y0 and x1 > x0:
                    small[i, j] = img_gray[y0:y1, x0:x1].mean()
                else:
                    small[i, j] = 0.0

    bits = (small >= threshold).astype(np.float32)
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


def phase_reconstruction(phase, g: int = 16, sigma: float = 0.6):
    """Return (recon_norm, recon_smoothed, threshold) for given phase vector.
    recon_norm is clipped to [0,1] and has shape (g,g).
    """
    if isinstance(phase, torch.Tensor):
        # ensure a 1-D numpy array
        arr = phase.detach().cpu().numpy().ravel()
    else:
        arr = np.asarray(phase).ravel()
    phi = np.clip(arr, -1, 1) * math.pi
    L = g * g
    if phi.size < L:
        phi = np.pad(phi, (0, L - phi.size))
    else:
        phi = phi[:L]
    spec = np.exp(1j * phi).reshape(g, g)

    # Enhanced smoothing with two passes
    try:
        from scipy.ndimage import gaussian_filter
        recon_smoothed = gaussian_filter(spec.real if np.iscomplexobj(spec) else spec, sigma=sigma)
        # Second pass with lower sigma to preserve details
        recon_smoothed = gaussian_filter(recon_smoothed, sigma=sigma/2)
    except Exception:
        try:
            from scipy.signal import convolve2d
            k = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
            k = k / k.sum()
            recon_smoothed = convolve2d(spec.real if np.iscomplexobj(spec) else spec, k, mode='same', boundary='symm')
            # Second pass
            recon_smoothed = convolve2d(recon_smoothed, k, mode='same', boundary='symm')
        except Exception:
            recon_smoothed = spec.real if np.iscomplexobj(spec) else spec

    # Enhanced normalization with robust scaling
    p10, p90 = np.percentile(recon_smoothed, [10, 90])
    if (p90 - p10) <= 1e-9:
        recon_norm = (recon_smoothed - recon_smoothed.min()) / (np.ptp(recon_smoothed) + 1e-9)
    else:
        recon_norm = np.clip((recon_smoothed - p10) / (p90 - p10), 0, 1)

    threshold = float(np.percentile(recon_norm, 60))  # Slightly increase threshold for clearer boundaries
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

    # Enhanced post-processing
    try:
        from scipy.ndimage import binary_closing, binary_opening, label
        struct = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=bool)  # Cross-shaped element
        
        # First closing to connect nearby components
        bits = binary_closing(bits.astype(bool), structure=struct).astype(np.float32)
        
        # Then opening to remove noise
        bits = binary_opening(bits.astype(bool), structure=struct).astype(np.float32)
        
        # Remove small components
        labeled, ncomp = label(bits.astype(bool))
        if ncomp > 0:
            counts = np.bincount(labeled.flatten())
            mask = np.zeros_like(bits, dtype=bool)
            min_size = 3  # Increase minimum component size
            for comp_id in range(1, counts.size):
                if counts[comp_id] >= min_size:
                    mask |= (labeled == comp_id)
            bits = mask.astype(np.float32)
    except Exception:
        pass

    return bits.flatten()


def kl_divergence(p, q):
    """Calculate Kullback-Leibler divergence between two probability distributions"""
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    
    # Normalize to make sure they sum to 1
    p = p / p.sum()
    q = q / q.sum()
    
    return np.sum(rel_entr(p, q))


def get_pixel_histogram(image, bins=10):
    """Calculate normalized histogram of pixel values"""
    hist, _ = np.histogram(image, bins=bins, range=(0, 1), density=True)
    return hist


def compute_moments(image):
    """Compute image moments (center of mass, size, orientation) without using ndimage.moments.
    Returns [x_cm, y_cm, area, theta]."""
    # Convert to binary if needed
    binary = image > 0.5 if image.dtype != bool else image
    mask = binary.astype(np.uint8)

    # Area
    area = float(mask.sum())
    if area == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Center of mass (y, x)
    try:
        y_cm, x_cm = ndimage.center_of_mass(mask)
    except Exception:
        # Fallback: compute manually
        ys, xs = np.nonzero(mask)
        y_cm = ys.mean() if ys.size > 0 else 0.0
        x_cm = xs.mean() if xs.size > 0 else 0.0

    # Compute covariance of coordinates for orientation
    ys, xs = np.nonzero(mask)
    coords = np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)
    if coords.shape[0] <= 1:
        theta = 0.0
    else:
        centered = coords - np.array([y_cm, x_cm], dtype=np.float32)
        cov = np.cov(centered, rowvar=False)
        # cov[0,0]=var_y, cov[1,1]=var_x, cov[0,1]=cov_yx
        mu20 = float(cov[0,0])
        mu02 = float(cov[1,1])
        mu11 = float(cov[0,1])
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) if (mu20 - mu02) != 0 or mu11 != 0 else 0.0

    return np.array([x_cm, y_cm, area, theta], dtype=np.float32)

def spectral_similarity(phase1, phase2):
    """Compare two phase patterns using spectral correlation"""
    # Normalize phases to [-pi, pi]
    p1 = np.clip(phase1, -np.pi, np.pi)
    p2 = np.clip(phase2, -np.pi, np.pi)
    
    # Compute complex exponentials
    z1 = np.exp(1j * p1)
    z2 = np.exp(1j * p2)
    
    # Calculate correlation
    corr = np.abs(np.sum(z1 * np.conj(z2))) / np.sqrt(np.sum(np.abs(z1)**2) * np.sum(np.abs(z2)**2))
    return float(corr)

class BitGridSensor:
    """Enhanced sensor with spectral pattern matching and moment features"""
    def __init__(self, dim: int = 128, grid: int = 16, threshold: float = 0.5, hist_bins: int = 10):
        self.dim = dim
        self.grid = grid
        self.threshold = threshold
        self.hist_bins = hist_bins
        self.cell_size = dim // grid
        
    def encode_image(self, img):
        """Encode image into bit grid representation using combined spectral and spatial features"""
        if isinstance(img, np.ndarray):
            img = img.copy()
        else:
            img = np.array(img)
            
        # Resize if needed
        if img.shape[0] != self.dim or img.shape[1] != self.dim:
            from PIL import Image
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((self.dim, self.dim))) / 255.0
            
        # Get bit grid directly (not recursively)
        bits = image_to_bitgrid(img, g=self.grid, threshold=self.threshold)
        phase = bitgrid_to_phase(bits, self.dim)
        recon, _, _ = phase_reconstruction(phase, g=self.grid)
        
        # Compute spatial moments
        moments = compute_moments(recon)
        
        # Compute histograms for Wasserstein distance
        hist_orig = get_pixel_histogram(img, bins=self.hist_bins)
        hist_recon = get_pixel_histogram(recon, bins=self.hist_bins)
        
        # Combine features
        features = {
            'bits': bits,
            'phase': phase.numpy(),
            'moments': moments,
            'hist_orig': hist_orig,
            'hist_recon': hist_recon,
            'reconstruction': recon
        }
        
        return features
    
    def compare_patterns(self, feat1, feat2):
        """Compare two encoded patterns using multiple metrics"""
        # Spectral similarity
        spec_sim = spectral_similarity(feat1['phase'], feat2['phase'])
        
        # Moment distance
        moment_dist = np.linalg.norm(feat1['moments'] - feat2['moments'])
        
        # Wasserstein distances for histograms
        w_dist_orig = wasserstein_distance(feat1['hist_orig'], feat2['hist_orig'])
        w_dist_recon = wasserstein_distance(feat1['hist_recon'], feat2['hist_recon'])
        
        # IoU for reconstructions
        intersection = np.sum(np.logical_and(feat1['reconstruction'] > 0.5, 
                                          feat2['reconstruction'] > 0.5))
        union = np.sum(np.logical_or(feat1['reconstruction'] > 0.5,
                                   feat2['reconstruction'] > 0.5))
        iou = intersection / (union + 1e-6)
        
        # Combined similarity score (weighted average)
        weights = {
            'spectral': 0.4,
            'moments': 0.2,
            'wasserstein': 0.2,
            'iou': 0.2
        }
        
        score = (weights['spectral'] * spec_sim +
                weights['moments'] * (1 - moment_dist) +
                weights['wasserstein'] * (1 - 0.5*(w_dist_orig + w_dist_recon)) +
                weights['iou'] * iou)
                
        return score, {
            'spectral_similarity': spec_sim,
            'moment_distance': moment_dist,
            'wasserstein_orig': w_dist_orig,
            'wasserstein_recon': w_dist_recon,
            'iou': iou,
            'combined_score': score
        }


def compute_histogram(data, bins=10, eps=1e-10):
    hist, _ = np.histogram(data, bins=bins, density=True)
    # Laplace smoothing
    hist = hist + eps
    hist = hist / hist.sum()  # Normalize
    return hist


def kl_divergence(p, q):
    # Compute KL divergence between two distributions
    return entropy(p, q)


def compare_distributions(pred_hist, mask_hist):
    # Returns similarity score (inverse of KL divergence)
    kl_div = kl_divergence(pred_hist, mask_hist)
    return 1.0 / (1.0 + kl_div)  # Convert to similarity score


class BitGridDecoder:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.scaler = StandardScaler()
        self.output_shape = None
    
    def fit(self, X, y):
        print('Computing training histograms...')
        # Store output shape for later use
        self.output_shape = y.shape[1] if len(y.shape) > 1 else y[0].size
        
        X_hist = np.array([compute_histogram(x, self.n_bins) for x in tqdm(X, desc='Processing input features')])
        y_hist = np.array([compute_histogram(y_i, self.n_bins) for y_i in tqdm(y, desc='Processing target values')])
        
        print('Scaling features...')
        X_hist = self.scaler.fit_transform(X_hist)
        
        # Store reference histograms and original data shapes
        self.ref_histograms = y_hist
        self.ref_data = y  # Store original training data
        return self
    
    def predict(self, X):
        print('Computing prediction histograms...')
        X_hist = np.array([compute_histogram(x, self.n_bins) for x in tqdm(X, desc='Processing inputs')])
        X_hist = self.scaler.transform(X_hist)
        
        print('Comparing distributions...')
        predictions = []
        for i, x_hist in enumerate(tqdm(X_hist, desc='Generating predictions')):
            similarities = [compare_distributions(x_hist, ref_hist) 
                          for ref_hist in self.ref_histograms]
            # Get index of most similar histogram
            most_similar_idx = np.argmax(similarities)
            # Use the corresponding original data as prediction
            pred = self.ref_data[most_similar_idx].reshape(-1)[:self.output_shape]
            predictions.append(pred)
        
        return np.array(predictions)


class MaskTemplate:
    """Simple container for a mask template and its descriptor."""
    def __init__(self, mask: np.ndarray, descriptor: np.ndarray, metadata: dict | None = None):
        # mask expected flat (grid*grid) with values {0,1}
        self.mask = mask.astype(np.uint8)
        self.descriptor = descriptor.astype(np.float32)
        self.metadata = metadata or {}


class TemplateStore:
    """Store and query mask templates. Templates are stored as masks (flattened) and compact descriptors.
    Descriptor is computed from a downsampled mask and an (optional) pixel histogram over the masked region.
    """
    def __init__(self, grid: int = 16, small: int = 8, hist_bins: int = 16):
        self.grid = grid
        self.small = small
        self.hist_bins = hist_bins
        self.templates: list[MaskTemplate] = []

    def _mask_to_descriptor(self, mask: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        """Compute compact descriptor for a mask (and optional image)."""
        # ensure flat
        mask = np.asarray(mask).reshape(-1)
        mask2d = mask.reshape(self.grid, self.grid)

        # downsample mask to small x small
        try:
            from PIL import Image
            img_mask = Image.fromarray((mask2d * 255).astype(np.uint8))
            img_small = img_mask.resize((self.small, self.small), resample=Image.BILINEAR)
            small_flat = (np.asarray(img_small).astype(np.float32) / 255.0).reshape(-1)
        except Exception:
            # fallback: simple pooling
            small_flat = mask2d.reshape(self.grid, self.grid)
            small_flat = small_flat.reshape(self.small, self.grid // self.small, self.small, self.grid // self.small).mean(axis=(1,3)).reshape(-1)

        # histogram of image pixels inside mask (if available)
        if image is not None:
            img = np.asarray(image).astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            # flatten grayscale or use first channel
            if img.ndim == 3:
                img_gray = img.mean(axis=2)
            else:
                img_gray = img
            mask_bool = mask.reshape(self.grid, self.grid).astype(bool)
            # resize image to grid to align
            try:
                from PIL import Image
                img_resized = np.array(Image.fromarray((img_gray * 255).astype(np.uint8)).resize((self.grid, self.grid))) / 255.0
            except Exception:
                img_resized = img_gray
            vals = img_resized[mask_bool]
            if vals.size == 0:
                hist = np.zeros(self.hist_bins, dtype=np.float32)
            else:
                hist, _ = np.histogram(vals, bins=self.hist_bins, range=(0, 1), density=False)
                hist = hist.astype(np.float32)
                hist = hist / (hist.sum() + 1e-9)
        else:
            hist = np.zeros(self.hist_bins, dtype=np.float32)

        desc = np.concatenate([small_flat, hist])
        # normalize descriptor for cosine similarity
        norm = np.linalg.norm(desc)
        if norm > 1e-9:
            desc = desc / norm
        return desc

    def add_template(self, mask: np.ndarray, image: np.ndarray | None = None, metadata: dict | None = None):
        """Add a new template. Mask must be shape (grid*grid,) or (grid,grid)."""
        m = np.asarray(mask).reshape(-1)
        if m.size != self.grid * self.grid:
            raise ValueError('Mask size does not match store grid')
        desc = self._mask_to_descriptor(m, image)
        tpl = MaskTemplate(m, desc, metadata)
        self.templates.append(tpl)
        return len(self.templates) - 1

    def match(self, image: np.ndarray, topk: int = 3) -> list[tuple[int, float]]:
        """Match an image (no mask) to stored templates. Returns list of (index, score) sorted desc.
        Score is cosine similarity in [0,1]."""
        if len(self.templates) == 0:
            return []
        # compute a wildcard descriptor by treating image as full mask candidate
        # create a pseudo-mask by thresholding image mean per cell
        try:
            # reuse BitGridSensor to get candidate mask
            sensor = BitGridSensor(dim=self.grid, grid=self.grid)
            candidate_mask = sensor.encode_image(image)
            candidate_mask = np.asarray(candidate_mask).reshape(-1)
        except Exception:
            # fallback: uniform unknown mask
            candidate_mask = np.ones(self.grid * self.grid, dtype=np.uint8)
        desc = self._mask_to_descriptor(candidate_mask, image)

        # build matrix of descriptors
        D = np.vstack([t.descriptor for t in self.templates])
        # cosine_similarity expects 2D
        sims = cosine_similarity(D, desc.reshape(1, -1)).reshape(-1)
        idxs = np.argsort(-sims)[:topk]
        return [(int(i), float(sims[i])) for i in idxs]

    def complete_mask(self, partial_mask: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        """Complete a partial mask using best matching template.
        partial_mask should use -1 for unknown cells, or NaN. Known values 0/1 are preserved.
        Returns completed mask as flat array of length grid*grid.
        """
        pm = np.asarray(partial_mask).reshape(-1).astype(np.float32)
        if pm.size != self.grid * self.grid:
            raise ValueError('Partial mask size mismatch')

        # find best match using available known cells: create a temporary descriptor weighted by known cells
        # create candidate mask by filling unknowns with zeros (neutral)
        known_mask = ~np.isnan(pm)
        if not np.any(known_mask):
            # no known info — return most common template
            if len(self.templates) == 0:
                return pm
            return self.templates[0].mask.copy()

        # score templates by agreement on known cells
        scores = []
        for t in self.templates:
            tmpl = t.mask.astype(np.float32)
            # agreement = proportion of equal bits among known cells
            agree = np.mean((tmpl[known_mask] == pm[known_mask]).astype(np.float32))
            scores.append(agree)
        best_idx = int(np.argmax(scores))
        best = self.templates[best_idx].mask.copy()

        # Merge: known values from pm take precedence, unknowns replaced from template
        completed = pm.copy()
        unknown = np.isnan(completed)
        completed[unknown] = best[unknown]
        # ensure binary
        completed = (completed >= 0.5).astype(np.uint8)
        return completed

    def save(self, path: str):
        joblib.dump({'grid': self.grid, 'small': self.small, 'hist_bins': self.hist_bins, 'templates': self.templates}, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        store = cls(grid=data.get('grid', 16), small=data.get('small', 8), hist_bins=data.get('hist_bins', 16))
        store.templates = data.get('templates', [])
        return store


# Convenience single-store for quick experiments
_TEMPLATE_STORE: TemplateStore | None = None

def get_template_store(grid: int = 16, small: int = 8, hist_bins: int = 16) -> TemplateStore:
    global _TEMPLATE_STORE
    if (_TEMPLATE_STORE is None):
        _TEMPLATE_STORE = TemplateStore(grid=grid, small=small, hist_bins=hist_bins)
    return _TEMPLATE_STORE

