import sys

sys.path.append("project_libs")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import copy
from plotting import plot_all, plot_phase_map
from functions import selu
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder

use_backprop = True
backprop_switched_off = False
baseline_weights = None

# WaveNetLayer
class WaveNetLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.i = nn.Parameter(torch.tensor(0.1))
        self.h = nn.Parameter(torch.tensor(0.01))
        self.prev_loss = None
        self.phase_kick_enabled = True
        self.phase_memory = []

    def forward(self, x, y_true, epoch=None):
        if use_backprop:
            return self.linear(x)

        with torch.no_grad():
            label_spacing = 2.5
            warmup = min(1.0, epoch / 50)
            raw_wave = torch.clamp(
                torch.tensor(math.pi, device=x.device)
                * x.float()
                * (y_true.float() * label_spacing)
                * self.h,
                -math.pi,
                math.pi,
            )
            wave_input = warmup * raw_wave + (1 - warmup) * 0.01 * torch.randn_like(x)
            wave_error = torch.sin(wave_input)
            wave_error = wave_error / (torch.norm(wave_error, dim=1, keepdim=True) + 1e-6)
            wave_error *= self.i
            wave_error = torch.clamp(wave_error, -1.0, 1.0)

            self.phase_memory.append(wave_error.clone())
            if len(self.phase_memory) > 10:
                self.phase_memory.pop(0)

            if len(self.phase_memory) >= 2:
                drift = (self.phase_memory[-1] - self.phase_memory[-2]).abs().mean().item()
                if drift < 1e-3:
                    self.i += 0.1 * torch.randn_like(self.i)
                    self.h += 0.1 * torch.randn_like(self.h)

            delta_w = torch.einsum("bi,bj->bij", wave_error, x)
            delta_b = wave_error.mean(dim=0)
            self.linear.weight += delta_w.mean(dim=0)
            self.linear.bias += delta_b
            self.linear.weight.data = torch.clamp(self.linear.weight.data, -0.5, 0.5)
            self.linear.bias.data = torch.clamp(self.linear.bias.data, -0.5, 0.5)

            # Resonator Feedback
            if epoch is not None and len(self.phase_memory) >= 2:
                previous = self.phase_memory[-2]
                current = self.phase_memory[-1]
                feedback = (previous - current) * 0.5
                self.linear.weight += torch.einsum("bi,bj->bij", feedback, x).mean(dim=0) * 0.002
                self.linear.bias += feedback.mean(dim=0) * 0.002

            # Reward-based boosting 
            target = torch.sin(torch.clamp(math.pi * x * (y_true * label_spacing) * self.h, -math.pi, math.pi))
            prediction = self.linear(x.float()).detach()
            phase_dist = (target - prediction).abs().mean(dim=1, keepdim=True)
            similarity = 1.0 - phase_dist
            reward_boost = (similarity > 0.90).float()
            if reward_boost.mean() > 0.01:
                boost = 0.02 * reward_boost.mean()
                self.i += boost
                self.h += boost
            print(f"[target phase]Target phase found-boosting")
            # Kick mechanisms
            if self.phase_kick_enabled and epoch is not None and epoch % 10 == 0:
                clarity = torch.std(wave_error.mean(dim=0)).item()
                if clarity < 0.01:
                    self.h += 0.01 * torch.randn_like(self.h)
                    self.i += 0.01 * torch.randn_like(self.i)

            # Adaptive noise
            epsilon = 0.02 if epoch < 10 else 0.002
            self.i += epsilon * torch.randn_like(self.i)
            self.h += epsilon * torch.randn_like(self.h)
            if epoch >= 500:
                self.i *= 0.99
                self.h *= 0.995

            self.i.data = torch.clamp(self.i.data, 0.03, 0.5)
            self.h.data = torch.clamp(self.h.data, 0.05, 0.1)

        return self.linear(x)


# Utilities 
def save_baseline(model):
    return copy.deepcopy(model.state_dict())

def log_phase_drift(current_weights, baseline_weights):
    total = 0.0
    for (_, param), (_, base) in zip(current_weights.items(), baseline_weights.items()):
        if param.data.ndim > 0:
            total += (param.data - base.data).abs().mean().item()
    return total

def maybe_switch_to_phase(epoch, loss_value, acc_value, model):
    global use_backprop, backprop_switched_off, baseline_weights
    if use_backprop and not backprop_switched_off:
        if epoch > 150 and loss_value < 3.5 and acc_value > 0.1:
            use_backprop = False
            backprop_switched_off = True
            baseline_weights = save_baseline(model)
            print(f"[SwitchMode] Epoch {epoch} — Backprop disabled, baseline saved")

def log_drift_from_baseline(epoch, model):
    if baseline_weights is not None:
        drift = log_phase_drift(model.state_dict(), baseline_weights)
        print(f"[PhaseDriftMap] Epoch {epoch} — Param drift from baseline: {drift:.4f}")

# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.SELU())
        self.output_layer = WaveNetLayer(hidden_size, output_size)
        self.wave_layers = nn.ModuleList([WaveNetLayer(hidden_size, hidden_size) for _ in range(num_layers)])
        self.functions = [selu, selu, selu]

    def forward(self, X, a, emotion_ids, emotion_labels, epoch=None):
        X = self.input_layer(X.float())
        Z_total = 0
        resonance = 0
        for i, wave_layer in enumerate(self.wave_layers):
            X = wave_layer(X + resonance, y_true=emotion_ids.unsqueeze(1).float(), epoch=epoch)
            func = self.functions[i % len(self.functions)]
            Z = torch.tensor(func(X.detach().cpu().numpy()), dtype=torch.float32, device=X.device)
            resonance = 0.8 * resonance + 0.2 * Z
            Z_total += Z
            X = torch.tanh(Z_total / len(self.wave_layers))
        return self.output_layer(X, emotion_ids.unsqueeze(1).float(), epoch), X

# Dataset Load
ds = load_dataset("Estwld/empathetic_dialogues_llm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

def embed_function(examples):
    with torch.no_grad():
        inputs = tokenizer([" ".join([c["content"] for c in conv]) for conv in examples["conversations"]],
                           padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        outputs = bert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return {"bert_embed": embeddings.numpy()}

ds = ds.map(embed_function, batched=True)
X = torch.tensor(ds["train"]["bert_embed"], dtype=torch.float32)

# Labels
label_encoder = LabelEncoder()
all_emotions = ds["train"]["emotion"]
label_encoder.fit(all_emotions)
ds = ds.map(lambda x: {"emotion": label_encoder.transform(x["emotion"])}, batched=True)
y = torch.tensor(ds["train"]["emotion"], dtype=torch.long)

# Setup
input_size = X.shape[1]
hidden_size = 32
output_size = len(label_encoder.classes_)
nn_model = NeuralNetwork(input_size, hidden_size, output_size, num_layers=20)

with torch.no_grad():
    for wave_layer in nn_model.wave_layers:
        wave_layer.i += 0.1 * torch.randn_like(wave_layer.i)
        wave_layer.h.data.fill_(1.0 / input_size)

class_weights = 0.5 / torch.bincount(y).float()
class_weights /= class_weights.sum()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=10000)
a = torch.full((X.size(0), 1), 0.5, dtype=torch.float32)

# Training
def train_model(nn_model, X, y, a, criterion, optimizer, scheduler=None, epochs=10000):
    loss_history, accuracy_history = [], []
    for epoch in range(epochs):
        if epoch % 300 == 0 and epoch < 1500:
            X += 0.05 * torch.randn_like(X)
            print(f"[PseudoInput] Epoch {epoch} — Mild stimulus applied")

        optimizer.zero_grad()
        output, phase = nn_model(X, a, y, y, epoch=epoch)
        loss = criterion(output, y)

        if use_backprop:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

        pred = torch.argmax(output, dim=1)
        acc = (pred == y).float().mean().item()
        loss_history.append(loss.item())
        accuracy_history.append(acc)

        with torch.no_grad():
            y_float = y.unsqueeze(1).float()
            wave_input = math.pi * X * y_float * nn_model.wave_layers[0].h
            phase_target = torch.sin(wave_input)
            pred_float = pred.unsqueeze(1).float()
            wave_pred_input = math.pi * X * pred_float * nn_model.wave_layers[0].h
            phase_pred = torch.sin(wave_pred_input)
            phase_distance = (phase_target - phase_pred).abs().mean().item()
            clarity = torch.std(phase.mean(dim=0)).item()

        maybe_switch_to_phase(epoch, loss.item(), acc, nn_model)
        log_drift_from_baseline(epoch, nn_model)

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc*100:.2f}%, PhaseDist: {phase_distance:.4f}, RCL: {clarity:.4f}")

    return loss_history, accuracy_history, output.detach()

loss_history, accuracy_history, final_output = train_model(nn_model, X, y, a, criterion, optimizer, scheduler)

plot_phase_map(X, y, nn_model.wave_layers[0].i.item(), nn_model.wave_layers[0].h.item(), title="Phase Error Map")
plot_all(y.numpy(), final_output.numpy(), loss_history, accuracy_history)
