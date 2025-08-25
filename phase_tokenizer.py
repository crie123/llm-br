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
        spectrum = np.fft.fft(codes)
        phase = np.angle(spectrum)
        phase = np.interp(phase, (-math.pi, math.pi), (-1, 1))
        phase = torch.tensor(phase, dtype=torch.float32)
        if phase.shape[0] < self.dim:
            pad = torch.zeros(self.dim - phase.shape[0])
            phase = torch.cat([phase, pad])
        else:
            phase = phase[:self.dim]
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
