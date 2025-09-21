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

'''
Future wave net structure(modern data types don't support such high prescision):

class WaveNetLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_classes=32):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.i = nn.Parameter(torch.tensor(0.1))
        self.h = nn.Parameter(torch.tensor(0.01))
        self.prev_loss = None
        self.phase_kick_enabled = True
        self.phase_memory = []
        self.emitter_memory = None
        self.last_acc = 0.0
        self.class_prototypes = generate_class_prototypes(n_classes, out_dim).to(self.linear.weight.device)

    def forward(self, x, y_true, epoch=None):
        if use_backprop:
            return self.linear(x)

        with torch.no_grad():
            label_spacing = 2.5
            raw_wave = torch.clamp(
                torch.tensor(math.pi, device=x.device)
                * x.float()
                * (y_true.float() * label_spacing)
                * self.h,
                -math.pi,
                math.pi,
            )
            wave_input = raw_wave
            base_wave = torch.sin(wave_input)
            base_wave = base_wave / (torch.norm(base_wave, dim=1, keepdim=True) + 1e-6)
            base_wave *= self.i
            base_wave = torch.clamp(base_wave, -1.0, 1.0)

            candidates = [base_wave]
            for angle in [-0.3, 0.3, -0.6, 0.6]:
                perturbed = base_wave + angle * torch.randn_like(base_wave) * 0.3
                candidates.append(torch.clamp(perturbed, -1.0, 1.0))

            best_error = base_wave
            best_score = -float("inf")
            for cand in candidates:
                delta_w = torch.einsum("bi,bj->bij", cand, x)
                delta_b = cand.mean(dim=0)
                temp_weight = self.linear.weight.data + delta_w.mean(dim=0)
                temp_bias = self.linear.bias.data + delta_b
                simulated = torch.matmul(x, temp_weight.T) + temp_bias
                pred = simulated.argmax(dim=1)
                acc = (pred == y_true.squeeze()).float().mean().item()
                if acc > best_score:
                    best_score = acc
                    best_error = cand

            wave_error = best_error

            if self.emitter_memory is not None:
                wave_error = 0.7 * wave_error + 0.3 * self.emitter_memory

            if len(self.phase_memory) > 3:
                avg_prev = sum(self.phase_memory[-3:]) / 3
                wave_error = 0.8 * wave_error + 0.2 * avg_prev

            self.phase_memory.append(wave_error.clone())
            if len(self.phase_memory) > 10:
                self.phase_memory.pop(0)

            if len(self.phase_memory) >= 2:
                drift = (self.phase_memory[-1] - self.phase_memory[-2]).abs().mean().item()
                if drift < 1e-3:
                    self.i += 0.1 * torch.randn_like(self.i)
                    self.h += 0.1 * torch.randn_like(self.h)

            # Phase Penalty
            target = torch.sin(torch.clamp(math.pi * x * (y_true * label_spacing) * self.h, -math.pi, math.pi))
            phase_penalty = (wave_error - target).abs().mean()
            wave_error -= 0.05 * phase_penalty

            # Phase Limiter 
            std = torch.std(wave_error, dim=1, keepdim=True)
            wave_error = torch.where(std > 0.5, wave_error * 0.5, wave_error)

            # Class Prototype Phase Similarity
            proto_target = self.class_prototypes[y_true.squeeze().long()]
            proto_sim = 1.0 - nn.functional.cosine_similarity(wave_error, proto_target, dim=1).mean()

            wave_error -= 0.02 * proto_sim  # attract to etalon

            delta_w = torch.einsum("bi,bj->bij", wave_error, x)
            delta_b = wave_error.mean(dim=0)
            self.linear.weight += delta_w.mean(dim=0)
            self.linear.bias += delta_b
            self.linear.weight.data = torch.clamp(self.linear.weight.data, -0.5, 0.5)
            self.linear.bias.data = torch.clamp(self.linear.bias.data, -0.5, 0.5)

            prediction = self.linear(x.float()).detach()
            phase_dist = (target - prediction).abs().mean(dim=1, keepdim=True)
            similarity = 1.0 - phase_dist
            if similarity.mean() > 0.92 and best_score > self.last_acc:
                self.emitter_memory = wave_error.clone().detach()
                self.last_acc = best_score

            if self.phase_kick_enabled and epoch is not None and epoch % 10 == 0:
                clarity = torch.std(wave_error.mean(dim=0)).item()
                if clarity < 0.01:
                    self.h += 0.01 * torch.randn_like(self.h)
                    self.i += 0.01 * torch.randn_like(self.i)

            epsilon = 0.02 if epoch < 10 else 0.002
            self.i += epsilon * torch.randn_like(self.i)
            self.h += epsilon * torch.randn_like(self.h)
            if epoch >= 500:
                self.i *= 0.99
                self.h *= 0.995

            self.i.data = torch.clamp(self.i.data, 0.03, 0.5)
            self.h.data = torch.clamp(self.h.data, 0.05, 0.1)

        return self.linear(x)
'''