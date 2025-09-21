import torch
import numpy as np
import math

class PhaseTokenizer:
    def __init__(self, dim=128, h=0.05, i=1.0):
        self.dim = dim
        self.h = h
        self.i = i

    def encode_array(self, arr: np.ndarray):
        base = torch.linspace(-1, 1, self.dim)
        mean_val = float(arr.mean())
        phase = self.i * torch.sin(math.pi * base * self.h * mean_val)
        return phase.unsqueeze(0)  # (1, dim)

    def encode_text(self, text: str):
        if text is None:
            text = ""
        codes = np.array([ord(c) for c in text], dtype=np.float32)
        if len(codes) == 0:
            codes = np.zeros(1, dtype=np.float32)

        # Base phase via FFT of character codes (as before)
        try:
            spectrum = np.fft.fft(codes)
            phase = np.angle(spectrum)
            phase = np.interp(phase, (-math.pi, math.pi), (-1, 1))
            base_phase = phase
        except Exception:
            base_phase = np.zeros_like(codes)

        # If input is short, augment with a deterministic char-position projection
        # to increase discriminability for short messages.
        if len(codes) < max(16, self.dim // 8):
            proj = np.zeros(self.dim, dtype=np.float32)
            for pos, c in enumerate(codes):
                idx = int((pos * 131 + int(c)) % self.dim)
                # small value depending on code and position
                proj[idx] += ((int(c) % 257) / 257.0) * (1.0 - pos / (len(codes) + 4))
            # smooth and normalize proj to [-1,1]
            if proj.max() - proj.min() > 1e-6:
                proj = (proj - proj.mean()) / (proj.std() + 1e-6)
            proj = np.tanh(proj)
            # combine base_phase (possibly shorter) with proj: tile or trim base_phase to dim
            if base_phase.shape[0] < self.dim:
                bp = np.concatenate([base_phase, np.zeros(self.dim - base_phase.shape[0])])
            else:
                bp = base_phase[:self.dim]
            combined = 0.7 * bp + 0.45 * proj
            # normalize to [-1,1]
            mx = np.max(np.abs(combined))
            if mx > 0:
                combined = combined / mx
            phase = combined
        else:
            # for longer inputs, pad or trim to dim
            if base_phase.shape[0] < self.dim:
                pad = np.zeros(self.dim - base_phase.shape[0])
                phase = np.concatenate([base_phase, pad])
            else:
                phase = base_phase[:self.dim]

        phase = torch.tensor(phase, dtype=torch.float32)
        return (self.i * phase).unsqueeze(0)

    def encode_image(self, img: np.ndarray):
        if img.ndim == 3:
            img = img.mean(axis=2)
        spectrum = np.fft.fft2(img)
        phase = np.angle(spectrum).flatten()
        phase = np.interp(phase, (-math.pi, math.pi), (-1, 1))
        phase = torch.tensor(phase, dtype=torch.float32)
        if phase.shape[0] < self.dim:
            pad = torch.zeros(self.dim - phase.shape[0])
            phase = torch.cat([phase, pad])
        else:
            phase = phase[:self.dim]
        return (self.i * phase).unsqueeze(0)
