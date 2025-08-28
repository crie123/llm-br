import sys

sys.path.append("project_libs")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import copy
from plotting import plot_phase_map, plot_consciousness_metrics, plot_resonant_clusters_matrix, plot_phase_resonance_field, plot_3d_phase_error_map
from functions import selu
from datasets import load_dataset
from phase_tokenizer import PhaseTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from bitgrid import BitGridSensor

use_backprop = True
backprop_switched_off = False
baseline_weights = None

class IcosPixyhArchive:
    def __init__(self, resolution=64, h=0.1, i=1.0, period=2 * math.pi):
        self.h = h
        self.i = i
        self.period = period
        self.resolution = resolution
        self.storage = {}

    def _coords(self, x, y):
        xi = int((x % self.period) / self.period * self.resolution)
        yi = int((y % self.period) / self.period * self.resolution)
        return xi, yi

    def write(self, x, y, value):
        xi, yi = self._coords(x, y)
        self.storage[(xi, yi)] = value.detach().cpu().clone() * self.i

    def read(self, x, y):
        xi, yi = self._coords(x, y)
        key = (xi, yi)
        if key in self.storage:
            return self.storage[key] / self.i
        # Exact cell is missing — signal absence explicitly
        return None

    def generate_wave_structure(self):
        grid = {}
        for xi in range(self.resolution):
            for yi in range(self.resolution):
                x = xi / self.resolution * self.period
                y = yi / self.resolution * self.period
                z = self.i * math.cos(math.pi * x * y * self.h)
                grid[(xi, yi)] = z
        return grid

def generate_class_prototypes(n_classes, dim, i=0.2, h=0.05):
    prototypes = []
    for c in range(n_classes):
        base = torch.linspace(-1, 1, dim)
        shift = (c / n_classes) * math.pi
        phase = i * torch.sin(math.pi * base * c * h + shift)
        prototypes.append(phase)
    return torch.stack(prototypes)

class EmpathicDatasetResponder:
    def __init__(self, dataset, label_encoder, bert_model, tokenizer, phase_dim=32, model=None, a_temp=None):
        self.entries = []
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.model = model
        self.a_temp = a_temp
        self.phase_dim = phase_dim

        print("[EmpathicResponder] Preparing dataset embeddings and phases...")

        def _extract_text_field(obj):
            if obj is None:
                return ""
            if isinstance(obj, str):
                return obj
            if isinstance(obj, list):
                parts = []
                for el in obj:
                    parts.append(_extract_text_field(el))
                return " ".join([p for p in parts if p])
            if isinstance(obj, dict):
                # try common keys
                for k in ("text", "utterance", "message", "content", "response", "utterances", "dialog"):
                    if k in obj and isinstance(obj[k], (str, list)):
                        return _extract_text_field(obj[k])
                # fallback to joining values
                return " ".join([_extract_text_field(v) for v in obj.values() if v])
            return str(obj)

        for example in dataset:
            # support dataset entry being a plain string or list as well
            if isinstance(example, str):
                text = ""
                response = example
                emotion = "unknown"
            elif isinstance(example, list):
                # list of turns -> join
                text = ""
                response = _extract_text_field(example)
                emotion = "unknown"
            elif isinstance(example, dict):
                # try multiple possible keys
                text = example.get("situation") or example.get("context") or example.get("dialog") or example.get("prompt") or ""
                if not text:
                    # try to glean context from other fields
                    text = _extract_text_field(example.get("context", ""))
                # responses may be under several keys
                if "conversations" in example:
                    response = example.get("conversations")
                elif "response" in example:
                    response = example.get("response")
                elif "utterances" in example:
                    response = example.get("utterances")
                elif "dialog" in example:
                    response = example.get("dialog")
                else:
                    # fallback: try common single-text fields
                    response = example.get("text") or example.get("message") or example.get("content") or ""
                emotion = example.get("emotion") or example.get("label") or "unknown"
                response = _extract_text_field(response)
                text = _extract_text_field(text)
            else:
                # unknown type
                continue

            # skip empty response
            if not response or str(response).strip() == "":
                continue

            # Choose a single assistant reply from possibly multi-turn conversation.
            def pick_response_text(resp):
                # strings -> assume single utterance
                if isinstance(resp, str):
                    return resp
                # list -> could be list of strings or list of turns (dicts)
                if isinstance(resp, list):
                    if len(resp) == 0:
                        return ""
                    if all(isinstance(x, str) for x in resp):
                        return resp[-1]
                    # list of dicts/turns: prefer last assistant turn if labeled
                    for turn in reversed(resp):
                        if isinstance(turn, dict):
                            role = turn.get('role') or turn.get('speaker')
                            if role and str(role).lower() in ('assistant', 'a', 'bot', 'system', 'agent'):
                                return _extract_text_field(turn.get('content') or turn.get('text') or turn)
                            # common content keys
                            for k in ('content', 'text', 'utterance', 'message'):
                                if k in turn and isinstance(turn[k], (str, list, dict)):
                                    return _extract_text_field(turn[k])
                    # fallback: last element
                    return _extract_text_field(resp[-1])
                # dict -> prefer assistant content if present
                if isinstance(resp, dict):
                    role = resp.get('role') or resp.get('speaker')
                    if role and str(role).lower() in ('assistant', 'a', 'bot', 'system', 'agent'):
                        return _extract_text_field(resp.get('content') or resp.get('text') or resp)
                    for k in ('content', 'text', 'utterance', 'message'):
                        if k in resp:
                            return _extract_text_field(resp[k])
                    return _extract_text_field(resp)
                return _extract_text_field(resp)

            resp_text = pick_response_text(response)

            # Encode response text into a deterministic phase embedding via tokenizer
            try:
                embed = self.tokenizer.encode_text(resp_text)  # (1, dim)
            except Exception:
                embed = torch.randn(1, self.tokenizer.dim)

            # ensure emotion label exists in encoder
            try:
                label_id_val = self.label_encoder.transform([emotion])[0]
            except Exception:
                # default to first class
                label_id_val = 0
            label_id = torch.tensor([label_id_val])
            with torch.no_grad():
                _, phase = self.model(embed, self.a_temp, label_id, label_id, epoch=999)

            self.entries.append({
                "context": text,
                "response": resp_text,
                "emotion": emotion,
                "phase": phase.squeeze(0).detach().cpu()
            })

        print(f"[EmpathicResponder] Loaded {len(self.entries)} responses with phases.")

    def reply_from_data(self, input_phase):
        input_phase = input_phase.detach().cpu()

        if not self.entries:
            raise ValueError("No entries loaded in EmpathicDatasetResponder.")

        similarities = [
            torch.nn.functional.cosine_similarity(input_phase, e["phase"], dim=0).item()
            for e in self.entries
        ]

        if not similarities:
            raise ValueError("No similarities computed — input phase may be invalid.")

        best_idx = int(np.argmax(similarities))
        best = self.entries[best_idx]
        return best["response"], f"[DATASET EMO: {best['emotion']}]"


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


# Utilities(switched off backpropagation, baseline saving, phase drift logging, maybe switch to phase mode)
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
        if epoch > 10 and loss_value < 3.5:
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
    
class SymbiontBridge(nn.Module):
    def __init__(self, main_layer, alpha=0.1, beta=0.01):
        super().__init__()
        self.main = main_layer
        self.alpha = alpha
        self.beta = beta

    def forward(self, external_phase):
        with torch.no_grad():
            # Modulate main layer's i and h based on external phase
            self.main.i += self.alpha * external_phase.mean()
            self.main.h += self.beta * torch.std(external_phase)
            print(f"[Symbiont] Modulated i/h with ext_phase: Δi={self.alpha * external_phase.mean().item():.4f}")

phase_archsim_history = []
drift_history = []
rcl_history = []
consciousness_score = []
cluster_assignments = []
phase_logs = []

if __name__ == '__main__':
    # Dataset Load
    ds = load_dataset("Estwld/empathetic_dialogues_llm")
    tokenizer = PhaseTokenizer(dim=128, h=0.05, i=1.0)
    bert_model = None  # Not used in wave-based pipeline; kept for backward compatibility

    def embed_function(examples):
        def _extract_text(obj):
            # Recursively extract strings from common dataset structures
            if isinstance(obj, str):
                return obj
            if isinstance(obj, dict):
                # common keys that may contain text
                for k in ("text", "utterance", "utterances", "message", "content", "situation", "transcript"):
                    v = obj.get(k) if k in obj else None
                    if isinstance(v, str):
                        return v
                    if isinstance(v, list) and all(isinstance(x, str) for x in v):
                        return " ".join(v)
                # fallback: take first string value
                for v in obj.values():
                    if isinstance(v, str):
                        return v
                    if isinstance(v, list) and all(isinstance(x, str) for x in v):
                        return " ".join(v)
                return str(obj)
            if isinstance(obj, list):
                parts = [_extract_text(x) for x in obj]
                return " ".join([p for p in parts if p])
            return str(obj)

        texts = []
        for conv in examples.get("conversations", []):
            try:
                txt = _extract_text(conv)
            except Exception:
                txt = str(conv)
            texts.append(txt)

        embeds = [tokenizer.encode_text(t).squeeze(0).numpy() for t in texts]
        return {"bert_embed": np.stack(embeds).astype(np.float32)}

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
    num_layers = 20
    nn_model = NeuralNetwork(input_size, hidden_size, output_size, num_layers=num_layers)

    a_temp = torch.full((1, 1), 0.5, dtype=torch.float32)

    # Augment training set with chatbot text responses and image-derived phases
    def _extract_text_simple(obj):
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for k in ("response", "utterance", "conversations", "dialog", "text", "content", "message", "prompt"):
                if k in obj:
                    v = obj[k]
                    if isinstance(v, str):
                        return v
                    if isinstance(v, list):
                        return " ".join([_extract_text_simple(x) for x in v if x])
            return " ".join([_extract_text_simple(v) for v in obj.values() if v])
        if isinstance(obj, list):
            return " ".join([_extract_text_simple(x) for x in obj])
        return str(obj)

    try:
        # Chat dataset augmentation (non-fatal if dataset missing)
        try:
            chat_ds = load_dataset("daily_dialog", split='train')
        except Exception:
            chat_ds = None
        if chat_ds is not None:
            max_chat = 500
            chat_small = chat_ds.select(list(range(min(len(chat_ds), max_chat))))
            chat_embs = []
            chat_labels = []
            for ex in chat_small:
                resp = _extract_text_simple(ex)
                if not resp or not str(resp).strip():
                    continue
                try:
                    emb = tokenizer.encode_text(resp).squeeze(0)
                except Exception:
                    emb = torch.randn(X.shape[1])
                chat_embs.append(emb)
                # attempt to map emotion if available, otherwise default to 0
                lab = 0
                try:
                    if isinstance(ex, dict) and 'emotion' in ex:
                        lab = int(label_encoder.transform([ex['emotion']])[0])
                except Exception:
                    lab = 0
                chat_labels.append(lab)
            if chat_embs:
                X = torch.cat([X, torch.stack(chat_embs)], dim=0)
                y = torch.cat([y, torch.tensor(chat_labels, dtype=torch.long)], dim=0)
            print(f"[Augment] Added {len(chat_embs)} chat examples to training set")
    except Exception as e:
        print(f"[Augment] chat augmentation failed: {e}")

    try:
        # Image-phase augmentation using BitGridSensor (non-fatal if dataset missing)
        sensor = BitGridSensor(dim=X.shape[1], grid=16, threshold=0.5)
        try:
            ds_imgs = load_dataset("cifar10", split="train[:200]")
        except Exception:
            ds_imgs = None
        img_embs = []
        img_labels = []
        if ds_imgs is not None:
            for item in ds_imgs:
                try:
                    img = np.asarray(item['img'].convert('L').resize((128, 128)), dtype=np.float32) / 255.0
                    emb = sensor.encode_image(img)
                    img_embs.append(emb)
                    img_labels.append(0)
                except Exception:
                    continue
        if img_embs:
            X = torch.cat([X, torch.stack(img_embs)], dim=0)
            y = torch.cat([y, torch.tensor(img_labels, dtype=torch.long)], dim=0)
        print(f"[Augment] Added {len(img_embs)} image examples to training set")
    except Exception as e:
        print(f"[Augment] image augmentation failed: {e}")

    # Archive and additional layers
    pixyh = IcosPixyhArchive(h=0.05, i=1.0)
    new_wave_layer = WaveNetLayer(32, 32)

    with torch.no_grad():
        for wave_layer in nn_model.wave_layers:
            wave_layer.i += 0.1 * torch.randn_like(wave_layer.i)
            wave_layer.h.data.fill_(1.0 / input_size)

    class_weights = 0.5 / torch.bincount(y).float()
    class_weights /= class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
    a = torch.full((X.size(0), 1), 0.5, dtype=torch.float32)

    # Training
    def train_model(nn_model, X, y, a, criterion, optimizer, scheduler=None, epochs=100):
        archetype_bank = generate_class_prototypes(output_size, 32).to(X.device)
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

            with torch.no_grad():
                pred = torch.argmax(output, dim=1)
                y_float = y.unsqueeze(1).float()
                wave_input = math.pi * X * y_float * nn_model.wave_layers[0].h
                phase_target = torch.sin(wave_input)
                pred_float = pred.unsqueeze(1).float()
                wave_pred_input = math.pi * X * pred_float * nn_model.wave_layers[0].h
                phase_pred = torch.sin(wave_pred_input)
                phase_distance = (phase_target - phase_pred).abs().mean().item()
                clarity = torch.std(phase.mean(dim=0)).item()

                drift = log_phase_drift(nn_model.state_dict(), baseline_weights) if baseline_weights else 0.0

                sim = nn.functional.cosine_similarity(phase, archetype_bank[y], dim=1).mean().item()
                score = sim * clarity * (1 - min(drift / 25, 1.0))

                phase_archsim_history.append(sim)
                rcl_history.append(clarity)
                drift_history.append(drift)
                consciousness_score.append(score)

                if epoch % 25 == 0:
                    km = KMeans(n_clusters=output_size, n_init='auto').fit(phase.cpu().numpy())
                    cluster_assignments.append(km.labels_)

            maybe_switch_to_phase(epoch, loss.item(), sim, nn_model)
            log_drift_from_baseline(epoch, nn_model)

            print(f"Epoch {epoch}, PhaseDist: {phase_distance:.4f}, ArchSim: {sim:.4f}, RCL: {clarity:.4f}, Drift: {drift:.4f}, Score: {score:.4f}")

            phase_logs.append({
                "epoch": epoch,
                "archsim": sim,
                "phasedist": phase_distance,
                "rcl": clarity,
                "drift": drift,
                "score": score
            })

            if epoch % 20 == 0 and not use_backprop and phase.shape[1] == 32:
                x_coord = X.mean().item()
                y_coord = y.float().mean().item()
                pixyh.write(x_coord, y_coord, phase)
                print(f"[PixyhArchive] Epoch {epoch} — Phase stored at ({x_coord:.2f}, {y_coord:.2f})")

            if epoch % 30 == 0 and not use_backprop and len(pixyh.storage) > 0:
                archived_phase = pixyh.read(x_coord, y_coord)
                if archived_phase is not None:
                    with torch.no_grad():
                        induced = new_wave_layer(archived_phase, y.unsqueeze(1).float(), epoch=epoch)
                        transferred = phase + 0.5 * induced
                        similarity = torch.nn.functional.cosine_similarity(
                            transferred.flatten(), archived_phase.flatten(), dim=0
                        ).item()
                        print(f"[TransPhase] Epoch {epoch} — CosSim to archive: {similarity:.4f}")

            if epoch % 40 == 0 and not use_backprop and len(pixyh.storage) > 0:
                external = pixyh.read(x_coord, y_coord)
                if external is not None:
                    symbiont = SymbiontBridge(nn_model.wave_layers[0])
                    symbiont(external)
        torch.save(phase_logs, "logs_epoch_phase_metrics.pt")
        torch.save({
        "model_state": nn_model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "num_layers": num_layers,
        "phase_dim": 32
        }, "pokrov_model.pt")

        return phase_archsim_history, rcl_history, drift_history, consciousness_score, cluster_assignments, output.detach()

    archsim, rcl, drift, score, clusters, final_output = train_model(nn_model, X, y, a, criterion, optimizer)

    plot_phase_map(X, y, nn_model.wave_layers[0].i.item(), nn_model.wave_layers[0].h.item(), title="Phase Error Map")
    plot_consciousness_metrics(archsim, rcl, drift, score)

    if clusters:
        last_clusters = clusters[-1]
        plot_resonant_clusters_matrix(y.numpy(), last_clusters)
        plot_phase_resonance_field(i=nn_model.wave_layers[0].i.item(),h=nn_model.wave_layers[0].h.item(),title="Phase Resonance Field")
        plot_3d_phase_error_map(i=nn_model.wave_layers[0].i.item(),h=nn_model.wave_layers[0].h.item(),title="3D Phase Error Map")