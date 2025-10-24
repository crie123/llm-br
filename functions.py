import numpy as np
import torch
import torch.nn as nn

# Function 4: Scaled Exponential Linear Unit (SELU)
def selu(x):
    return torch.nn.functional.selu(torch.tensor(x)).numpy()

# FlexiblePReLU: Parameterized ReLU with emotion-based sector weights(legacy)
class FlexiblePReLU(nn.Module):
    def __init__(self, sector_weights):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(0.09, dtype=torch.float32))
        self.sector_weights = sector_weights

    def forward(self, x, a, emotion_labels):
        lower = a - 0.5
        upper = a + 0.5
        in_region = (a >= -1) & (a <= 1) & (x > lower) & (x < upper)

        emotion_multipliers = torch.tensor(
            [self.sector_weights.get(emotion_to_sector.get(em, ''), 1.0) for em in emotion_labels],
            dtype=torch.float32,
            device=x.device
        ).unsqueeze(1)

        out = torch.where(in_region, x * self.r * emotion_multipliers, x)
        return out
    
# Emotions and their corresponding character sectors(legacy)
emotion_to_sector = {
    "afraid": "melancholic", "angry": "choleric", "annoyed": "choleric",
    "anticipating": "sanguine", "anxious": "melancholic", "apprehensive": "melancholic",
    "confident": "sanguine", "content": "phlegmatic", "devastated": "melancholic",
    "disappointed": "melancholic", "embarrassed": "melancholic", "excited": "sanguine",
    "faithful": "phlegmatic", "grateful": "phlegmatic", "guilty": "melancholic",
    "hopeful": "sanguine", "impressed": "sanguine", "jealous": "choleric",
    "joyful": "sanguine", "lonely": "melancholic", "nostalgic": "phlegmatic",
    "proud": "sanguine", "sad": "melancholic", "terrified": "melancholic",
    "trusting": "phlegmatic"
}

sector_to_weight = {
    "melancholic": 0.92,
    "choleric": 1.08,
    "phlegmatic": 0.98,
    "sanguine": 1.05,
}
    
# Function 1: Exponentially decreases then increases (symmetric tanh)(legacy)
def f1(x):
    return -np.tanh(2 * x)

# Function 2: Exponential rise and fall, then linear(legacy)
def f2(x):
    return np.piecewise(x, [x <= 1, x > 1], [lambda x: np.tanh(2 * x), lambda x: 0.07 * x + 0.894])

# Function 3: Linearly decreases and increases, smoothed with tanh near joins(legacy)
def f3(x):
    return np.piecewise(x, [x <= 0, x > 0], [lambda x: -0.5 * x - 0.5 * np.tanh(3 * (x + 0.5)), lambda x: 0.5 * x - 0.5 * np.tanh(3 * (x - 0.5))])
