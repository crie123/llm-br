# evaluate_wave_model.py
import os
import sys

# Add project_libs to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "project_libs"))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Any

# Torch + sklearn
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# Import from project_libs
from neural_network import NeuralNetwork
from phase_tokenizer import PhaseTokenizer
from project_libs import WaveNetwork

# Try to import user's model and tokenizer (robustly)
MODEL_CKPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pokrov_model.pt")
REPORT_DIR = "eval_report"
os.makedirs(REPORT_DIR, exist_ok=True)

# --- Utilities ----------------------------------------------------------------
def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[WARN] cannot load checkpoint {path}: {e}")
        return None

def ensure_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = ensure_device()
print(f"[ENV] device = {DEVICE}")

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def cos_sim(a, b):
    # a, b are 1D numpy arrays or tensors
    a_t = torch.tensor(a, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32)
    return float(F.cosine_similarity(a_t.unsqueeze(0), b_t.unsqueeze(0)).item())

# --- Load user's model class --------------------------------------------------
try:
    from neural_network import NeuralNetwork
    print("[INFO] imported NeuralNetwork from neural_network.py")
except Exception as e:
    print(f"[WARN] failed to import NeuralNetwork: {e}")
    NeuralNetwork = None

# Build or load model 
MODEL_INPUT_DIM = 128  # Changed back to 128 to match saved weights
MODEL_HIDDEN = 32 
MODEL_OUTPUT = 32
MODEL_LAYERS = 20

def build_model():
    if NeuralNetwork is None:
        # Minimal fallback model (toy) — same interface: forward(X,a,emotion_ids,emotion_labels,epoch)
        class DummyModel(torch.nn.Module):
            def __init__(self, input_size=MODEL_INPUT_DIM, hidden=32, output=32):
                super().__init__()
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden, output)
                )
                self.feature_dim = input_size
                # fake wave layers attribute for compatibility if used
                self.wave_layers = torch.nn.ModuleList()
            def forward(self, X, a_temp=None, em1=None, em2=None, epoch=None):
                # X: [B, D]
                logits = self.fc(X.float())
                # create a fake phase vector from logits (bounded)
                phase = torch.tanh(torch.randn(X.shape[0], MODEL_INPUT_DIM) * 0.1 + logits.unsqueeze(1)[:,0:1])
                return logits, phase
        print("[INFO] Using DummyModel: behaviour approx.")
        return DummyModel(input_size=MODEL_INPUT_DIM, hidden=MODEL_HIDDEN, output=MODEL_OUTPUT)
    else:
        model = NeuralNetwork(input_size=MODEL_INPUT_DIM, hidden_size=MODEL_HIDDEN,
                              output_size=MODEL_OUTPUT, num_layers=MODEL_LAYERS)
        # try to load checkpoint
        if os.path.exists(MODEL_CKPT):
            ck = safe_torch_load(MODEL_CKPT)
            if ck is not None:
                try:
                    if "model_state" in ck:
                        model.load_state_dict(ck["model_state"])
                        print(f"[INFO] Loaded model_state from {MODEL_CKPT}")
                    else:
                        # maybe state_dict directly
                        model.load_state_dict(ck)
                        print(f"[INFO] Loaded state_dict from {MODEL_CKPT}")
                except Exception as e:
                    print(f"[WARN] Failed to load weights: {e}. Using random init.")
        else:
            print(f"[INFO] Checkpoint {MODEL_CKPT} not found — using random init.")
        return model

model = build_model()
model.to(DEVICE)
model.eval()

# Try to get phi dimension (phase dim) from model pipeline. We'll assume tokenizer.dim or 128 default
PHASE_DIM = MODEL_INPUT_DIM
try:
    # if wave layers exist and have class_prototypes with shape
    if hasattr(model, 'wave_layers') and len(model.wave_layers) > 0:
        PHASE_DIM = getattr(model.wave_layers[0], 'class_prototypes').shape[1]
        print("[INFO] inferred PHASE_DIM from model:", PHASE_DIM)
except Exception:
    pass

# --- Tokenizer / PhaseTokenizer fallback -------------------------------------
try:
    from phase_tokenizer import PhaseTokenizer
    tokenizer = PhaseTokenizer(dim=PHASE_DIM, h=0.05, i=1.0)
    print("[INFO] PhaseTokenizer loaded.")
except Exception as e:
    tokenizer = None
    print(f"[WARN] PhaseTokenizer not found, will use random phase embeddings. ({e})")

# helper a_temp (attention temp) as in your code
a_temp = torch.full((1, 1), 0.5, dtype=torch.float32)

# --- Synthetic dataset T2 generation (P1-b paraphrases) ---------------------
EMOTIONS = [f"emo_{i}" for i in range(MODEL_OUTPUT)]  # 32 emotions by name
NUM_BASE = 5
NUM_PARA = 3  # P1-b style paraphrases per base
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# small template pools for base and paraphrases (moderate rephrasing)
BASE_TEMPLATES = [
    "I feel {} about this.",
    "This makes me feel {}.",
    "Today I'm feeling {}.",
    "It's a {} kind of day.",
    "I'm experiencing {} feelings."
]
PARAPHRASE_TEMPLATES = [
    "I am kind of {} today.",
    "I'm feeling quite {} right now.",
    "There's a {} vibe happening.",
    "My mood is sort of {}.",
    "It kind of feels {} to me."
]

# Simple mapping of abstract "emotion words" — these are placeholders and do not need to be human-accurate
SENTIMENT_WORDS = [
    "joyful", "angry", "sad", "surprised", "calm", "anxious", "hopeful", "bored",
    "curious", "disgusted", "proud", "ashamed", "lonely", "content", "confused", "excited",
    "fearful", "relieved", "guilty", "determined", "tired", "amused", "nostalgic", "frustrated",
    "trusting", "embarrassed", "jealous", "satisfied", "moved", "optimistic", "pensive", "apathetic"
]
# ensure length
if len(SENTIMENT_WORDS) < len(EMOTIONS):
    # fill
    SENTIMENT_WORDS = SENTIMENT_WORDS + ["neutral"] * (len(EMOTIONS) - len(SENTIMENT_WORDS))

# Build dataset as list of dicts: {emotion_idx, base_text, paraphrases}
dataset_examples = []
for ei in range(len(EMOTIONS)):
    emo_word = SENTIMENT_WORDS[ei % len(SENTIMENT_WORDS)]
    for b in range(NUM_BASE):
        base = random.choice(BASE_TEMPLATES).format(emo_word)
        paras = []
        for p in range(NUM_PARA):
            paras.append(random.choice(PARAPHRASE_TEMPLATES).format(emo_word))
        dataset_examples.append({
            "emotion_idx": ei,
            "base": base,
            "paraphrases": paras
        })
print(f"[DATA] Built synthetic dataset: {len(dataset_examples)} examples (T2)")

# Flatten list for inference items: keep mapping to (emotion, text, kind, base_id)
inference_items = []
for i, ex in enumerate(dataset_examples):
    base_id = i
    inference_items.append({"emotion": ex["emotion_idx"], "text": ex["base"], "kind": "base", "base_id": base_id})
    for pi, ptext in enumerate(ex["paraphrases"]):
        inference_items.append({"emotion": ex["emotion_idx"], "text": ptext, "kind": "para", "base_id": base_id, "para_id": pi})

# --- Phase embedding for text (using tokenizer if available, else approximate) ---
def encode_text_to_phase(text: str) -> torch.Tensor:
    """
    Return phase embedding vector in [-1,1] range of length PHASE_DIM.
    If tokenizer exists, call tokenizer.encode_text(text). Otherwise, produce a deterministic pseudo embedding.
    """
    if tokenizer is not None:
        try:
            with torch.no_grad():
                enc = tokenizer.encode_text(text)  # expected shape [1, dim]
                if isinstance(enc, torch.Tensor):
                    v = enc.squeeze(0)
                else:
                    v = torch.tensor(enc).squeeze(0)
                # clamp to [-1,1]
                v = torch.tanh(v.float())
                # ensure length
                if v.numel() != PHASE_DIM:
                    v = F.pad(v, (0, max(0, PHASE_DIM - v.numel())))
                    v = v[:PHASE_DIM]
                return v
        except Exception as e:
            print(f"[WARN] tokenizer failed for text -> fallback: {e}")

    # deterministic pseudo-phase: hash text -> vector
    h = abs(hash(text)) % (10**8)
    rng = np.random.RandomState(h)
    v = rng.normal(loc=0.0, scale=0.6, size=(PHASE_DIM,))
    v = np.tanh(v).astype(np.float32)
    return torch.tensor(v, dtype=torch.float32)

# --- REF-B1: single reference phase (fixed continuous function) -------------
# REF-B1 chosen: single universal phase reference for IA2
REF_I = 1.0
REF_H = 0.05

def build_ref_phase(dim: int, i: float = REF_I, h: float = REF_H) -> torch.Tensor:
    """
    Build single reference phase vector of length dim using continuous function.
    Use a simple separable function: phi[k] = i * sin(pi * (k/dim) * ((k+1)/dim) * h)
    then normalized to [-1,1].
    """
    ks = np.arange(dim, dtype=np.float64)
    xs = (ks / max(1, dim-1))
    ys = ((ks + 1) / max(1, dim))
    phi = i * np.sin(np.pi * xs * ys * h)
    # scale to [-1,1] (already small), but ensure numeric stability
    max_abs = max(1e-9, np.max(np.abs(phi)))
    phi_norm = phi / max_abs
    phi_clipped = np.clip(phi_norm, -1.0, 1.0).astype(np.float32)
    return torch.tensor(phi_clipped, dtype=torch.float32)

# Try to get phase dimension from model
try:
    if hasattr(model, 'wave_net'):
        PHASE_DIM = model.wave_net.phase_dim
    elif hasattr(model, 'wave_layers') and len(model.wave_layers) > 0:
        PHASE_DIM = getattr(model.wave_layers[0], 'class_prototypes').shape[1]
    print("[INFO] inferred PHASE_DIM from model:", PHASE_DIM)
except Exception:
    PHASE_DIM = 32  # Default to 32 if can't infer
    print("[INFO] using default PHASE_DIM:", PHASE_DIM)

REF_PHASE = build_ref_phase(PHASE_DIM)
print(f"[REF] built reference phase (REF-B1) dim={PHASE_DIM}")

# --- Inference wrapper (use model forward exactly like user's loop) ----------
def model_infer_phase(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (logits_np, phase_np)
    Uses encode_text_to_phase to create embedding, then calls model.
    Tries to mimic model(phase_input, a_temp, dummy_label, dummy_label, epoch=999)
    """
    phase_input = encode_text_to_phase(text).unsqueeze(0).to(DEVICE)  # [1, PHASE_DIM]
    # Create proper emotion_ids tensor
    dummy_label = torch.tensor([0], dtype=torch.long, device=DEVICE)
    # Create proper attention temp tensor
    a = torch.full((1, 1), 0.5, dtype=torch.float32, device=DEVICE)
    
    with torch.no_grad():
        try:
            # Reshape input if needed
            if phase_input.shape[1] != MODEL_INPUT_DIM:
                phase_input = F.pad(phase_input, (0, MODEL_INPUT_DIM - phase_input.shape[1])) \
                            if phase_input.shape[1] < MODEL_INPUT_DIM \
                            else phase_input[:, :MODEL_INPUT_DIM]
            
            # Call with all required arguments
            logits, phase = model(phase_input, a, dummy_label, dummy_label, epoch=999)
            
            # Convert phase to 1D vector with range [-1,1] if necessary
            if isinstance(phase, torch.Tensor):
                p = phase.squeeze(0).detach().cpu().float().numpy()
            else:
                p = np.array(phase).squeeze(0)
            l = logits.squeeze(0).detach().cpu().numpy()
            return l, p
        except Exception as e:
            # Log the error for debugging
            print(f"[ERR] model forward failed: {e}")
            # Return random outputs as fallback
            l = np.random.randn(MODEL_OUTPUT)
            p = np.tanh(np.random.randn(PHASE_DIM)).astype(np.float32)
            return l, p

# --- Run inference on the full synthetic set ---------------------------------
print("[RUN] Running inference on synthetic T2 dataset...")
start = time.time()
results = []
for it in inference_items:
    logits, phase = model_infer_phase(it["text"])
    pred_class = int(np.argmax(logits))
    results.append({
        "text": it["text"],
        "true_emotion": int(it["emotion"]),
        "pred_class": pred_class,
        "phase": phase.astype(np.float32),
        "kind": it.get("kind", "base"),
        "base_id": it.get("base_id", -1)
    })
end = time.time()
print(f"[RUN] Inference finished: {len(results)} items, time {end-start:.2f}s")

# --- Group results per base example / emotion --------------------------------
by_base = defaultdict(list)
by_emotion = defaultdict(list)
for r in results:
    by_base[r["base_id"]].append(r)
    by_emotion[r["true_emotion"]].append(r)

# --- Metrics calculations ----------------------------------------------------
def compute_semantic_stability(by_base_map: Dict[int, List[dict]]) -> Tuple[float, Dict[int,float]]:
    """
    For each base (one base + its paraphrases), measure fraction of paraphrases matching base's predicted class.
    Returns global average and per-emotion map aggregated.
    """
    per_base_scores = {}
    per_emotion_scores = defaultdict(list)
    for base_id, items in by_base_map.items():
        # find base item
        base_item = next((x for x in items if x["kind"]=="base"), items[0])
        base_pred = base_item["pred_class"]
        # consider paraphrases
        paras = [x for x in items if x["kind"]=="para"]
        if not paras:
            # single item -> score 1.0
            score = 1.0
        else:
            same = sum(1 for p in paras if p["pred_class"]==base_pred)
            score = same / max(1, len(paras))
        per_base_scores[base_id] = score
        per_emotion_scores[base_item["true_emotion"]].append(score)
    # average per-base
    overall = float(np.mean(list(per_base_scores.values()))) if per_base_scores else 0.0
    per_emotion_avg = {emo: float(np.mean(v)) if v else 0.0 for emo,v in per_emotion_scores.items()}
    return overall, per_emotion_avg

def compute_phase_cohesion(by_emotion_map: Dict[int, List[dict]]) -> Tuple[float, Dict[int,float]]:
    """
    Phase cohesion: average pairwise cosine similarity among phases within same emotion.
    """
    per_em = {}
    all_vals = []
    for emo, items in by_emotion_map.items():
        vecs = np.stack([it["phase"] for it in items], axis=0)
        if vecs.shape[0] <= 1:
            per_em[emo] = 1.0
            all_vals.append(1.0)
            continue
        # compute pairwise cosines
        sims = []
        for i in range(vecs.shape[0]):
            for j in range(i+1, vecs.shape[0]):
                sims.append(F.cosine_similarity(torch.tensor(vecs[i]).unsqueeze(0),
                                               torch.tensor(vecs[j]).unsqueeze(0)).item())
        mean_sim = float(np.mean(sims)) if sims else 0.0
        per_em[emo] = mean_sim
        all_vals.append(mean_sim)
    overall = float(np.mean(all_vals)) if all_vals else 0.0
    return overall, per_em

def compute_spread_S2(by_emotion_map: Dict[int, List[dict]], n_clusters=4) -> Tuple[float, Dict[int,float]]:
    """
    S2: for each emotion, cluster its phases into n_clusters, compute share of largest cluster;
    spread = 1 - max_cluster_share (so 0 if all collapse into one cluster).
    """
    per_em = {}
    for emo, items in by_emotion_map.items():
        vecs = np.stack([it["phase"] for it in items], axis=0)
        if vecs.shape[0] <= 1:
            per_em[emo] = 0.0
            continue
        k = min(n_clusters, vecs.shape[0])
        try:
            km = KMeans(n_clusters=k, random_state=SEED).fit(vecs)
            labels = km.labels_
            counts = np.bincount(labels)
            max_share = float(counts.max()) / float(len(labels))
            spread = 1.0 - max_share
            per_em[emo] = spread
        except Exception as e:
            per_em[emo] = 0.0
    overall = float(np.mean(list(per_em.values()))) if per_em else 0.0
    return overall, per_em

def compute_cluster_purity(results_list: List[dict], n_clusters=None) -> float:
    """
    Cluster across all phases into n_clusters (default = number of emotions),
    assign predicted majority emotion to each cluster, compute purity.
    """
    vecs = np.stack([r["phase"] for r in results_list], axis=0)
    true_labels = np.array([r["true_emotion"] for r in results_list], dtype=int)
    num_emotions = len(set(true_labels))
    k = n_clusters or num_emotions
    k = min(k, vecs.shape[0])
    try:
        km = KMeans(n_clusters=k, random_state=SEED).fit(vecs)
        cl_labels = km.labels_
        purity_sum = 0
        for c in range(k):
            idxs = np.where(cl_labels == c)[0]
            if idxs.size == 0:
                continue
            true_in_cluster = true_labels[idxs]
            # majority
            vals, counts = np.unique(true_in_cluster, return_counts=True)
            majority = counts.max()
            purity_sum += majority
        purity = float(purity_sum) / float(len(vecs))
    except Exception as e:
        print(f"[WARN] cluster purity calculation failed: {e}")
        purity = 0.0
    return purity

def compute_neuro_consistency(results_list: List[dict], repeats=10) -> Tuple[float, Dict[int,float]]:
    """
    Neuro-consistency approximated by repeated inference on the same text (no input change).
    We'll repeat inference `repeats` times for a small sample subset and compute phase cosine stability.
    """
    sample = results_list[::max(1, len(results_list)//80)]  # sample up to ~80 points
    per_idx = {}
    for s in sample:
        text = s["text"]
        phases = []
        for _ in range(repeats):
            _, p = model_infer_phase(text)
            phases.append(p)
        phases = np.stack(phases, axis=0)  # [R, dim]
        # compute mean pairwise cosine
        sims = []
        for i in range(phases.shape[0]):
            for j in range(i+1, phases.shape[0]):
                sims.append(F.cosine_similarity(torch.tensor(phases[i]).unsqueeze(0),
                                               torch.tensor(phases[j]).unsqueeze(0)).item())
        per_idx[text] = float(np.mean(sims)) if sims else 1.0
    # aggregate by emotion
    per_em = defaultdict(list)
    for r in results_list:
        if r["text"] in per_idx:
            per_em[r["true_emotion"]].append(per_idx[r["text"]])
    per_em_mean = {emo: float(np.mean(v)) if v else 1.0 for emo, v in per_em.items()}
    overall = float(np.mean(list(per_em_mean.values()))) if per_em_mean else 1.0
    return overall, per_em_mean

def compute_IA2_phase_alignment(results_list: List[dict], ref_phase: torch.Tensor) -> Tuple[float, Dict[int,float]]:
    """
    IA2 = mean cosine similarity between model phase and reference phase (complex-phase metric)
    We'll use cosine similarity on unit complex exponentials: mean(Re(exp(j*(phi_model - phi_ref))))
    """
    ref = ref_phase.cpu().numpy()
    per_em = defaultdict(list)
    for r in results_list:
        m = r["phase"]
        # complex dot mean
        # compute mean real part of e^{j*(m - ref)}
        delta = m - ref
        z = np.cos(delta)  # real part of complex exponential mean approximated by mean(cos)
        val = float(np.mean(z))
        per_em[r["true_emotion"]].append(val)
    per_em_mean = {emo: float(np.mean(v)) if v else 0.0 for emo, v in per_em.items()}
    overall = float(np.mean(list(per_em_mean.values()))) if per_em_mean else 0.0
    return overall, per_em_mean

# Compute metrics
print("[METRICS] computing metrics...")
sem_overall, sem_per_em = compute_semantic_stability(by_base)
cohesion_overall, cohesion_per_em = compute_phase_cohesion(by_emotion)
spread_overall, spread_per_em = compute_spread_S2(by_emotion, n_clusters=4)
purity = compute_cluster_purity(results)
neuro_overall, neuro_per_em = compute_neuro_consistency(results, repeats=6)
ia2_overall, ia2_per_em = compute_IA2_phase_alignment(results, REF_PHASE)

# Cognitive Integrity Index (CII) as defined earlier
# CII = 0.25*Purity + 0.25*Paraphrase + 0.2*NeuroConsistency + 0.15*Superposition + 0.1*MemoryRetention + eps
# We don't compute MemoryRetention here explicitly; approximate with cohesion (proxy)
memory_retention_proxy = cohesion_overall
CII = 0.25 * purity + 0.25 * sem_overall + 0.2 * neuro_overall + 0.15 * spread_overall + 0.1 * memory_retention_proxy
CII = float(np.clip(CII, 0.0, 1.0))

# --- Save numeric summary ----------------------------------------------------
summary = {
    "dataset": "synthetic_T2",
    "n_examples": len(results),
    "metrics": {
        "semantic_stability_overall": sem_overall,
        "phase_cohesion_overall": cohesion_overall,
        "spread_S2_overall": spread_overall,
        "cluster_purity": purity,
        "neuro_consistency_overall": neuro_overall,
        "ia2_overall": ia2_overall,
        "CII": CII
    }
}
with open(os.path.join(REPORT_DIR, "final_metrics.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print("[SAVE] final_metrics.json saved.")

# --- Per-emotion table -------------------------------------------------------
emotion_table = []
for emo_idx in sorted(by_emotion.keys()):
    emotion_table.append({
        "emotion_idx": int(emo_idx),
        "semantic_stability": float(sem_per_em.get(emo_idx, 0.0)),
        "phase_cohesion": float(cohesion_per_em.get(emo_idx, 0.0)),
        "spread_S2": float(spread_per_em.get(emo_idx, 0.0)),
        "neuro_consistency": float(neuro_per_em.get(emo_idx, 1.0)),
        "ia2": float(ia2_per_em.get(emo_idx, 0.0))
    })
with open(os.path.join(REPORT_DIR, "per_emotion_metrics.json"), "w", encoding="utf-8") as fh:
    json.dump(emotion_table, fh, indent=2)
print("[SAVE] per_emotion_metrics.json saved.")

# --- Diagnostic plots --------------------------------------------------------
def plot_histogram_metric(name: str, values: List[float], out_file: str):
    plt.figure(figsize=(6,3))
    plt.hist(values, bins=20, density=False)
    plt.title(name)
    plt.xlabel(name)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

# Phase UMAP requires umap-learn; fallback to PCA if not available
def embed_to_2d(phases: np.ndarray):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=SEED)
        emb = reducer.fit_transform(phases)
        return emb
    except Exception:
        # PCA fallback
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=SEED)
        emb = pca.fit_transform(phases)
        return emb

# Prepare arrays
all_phases = np.stack([r["phase"] for r in results], axis=0)
all_true = np.array([r["true_emotion"] for r in results], dtype=int)
all_preds = np.array([r["pred_class"] for r in results], dtype=int)

# UMAP/PCA
emb2 = embed_to_2d(all_phases)
plt.figure(figsize=(6,5))
plt.scatter(emb2[:,0], emb2[:,1], c=all_true, cmap='Spectral', s=6)
plt.title("Phase field (colored by true emotion)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "phase_field_umap.png"))
plt.close()

# Confusion matrix between true_emotion and predicted class
try:
    cm = confusion_matrix(all_true, all_preds, labels=list(range(MODEL_OUTPUT)))
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("Confusion matrix (true vs pred)")
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"[WARN] confusion plot failed: {e}")

# Per-metric histograms
plot_histogram_metric("semantic_stability (per-base)", list(sem_per_em.values()), os.path.join(REPORT_DIR, "semantic_stability_hist.png"))
plot_histogram_metric("phase_cohesion (per-emotion)", list(cohesion_per_em.values()), os.path.join(REPORT_DIR, "phase_cohesion_hist.png"))
plot_histogram_metric("spread_S2 (per-emotion)", list(spread_per_em.values()), os.path.join(REPORT_DIR, "spread_S2_hist.png"))
plot_histogram_metric("ia2 (per-emotion)", list(ia2_per_em.values()), os.path.join(REPORT_DIR, "ia2_hist.png"))

# Save sample predictions CSV
import csv
with open(os.path.join(REPORT_DIR, "predictions.csv"), "w", encoding="utf-8", newline='') as fh:
    w = csv.writer(fh)
    w.writerow(["text", "true_emotion", "pred_class", "kind", "base_id"])
    for r in results:
        w.writerow([r["text"].replace("\n", " "), r["true_emotion"], r["pred_class"], r["kind"], r["base_id"]])
print("[SAVE] predictions.csv saved.")

# --- Print summary to console -----------------------------------------------
print("\n=== FINAL EVALUATION (Pokrov Cognition Report) ===")
print(f"Dataset (synthetic T2) examples: {len(results)}")
print(f"Semantic Stability (paraphrase consistency): {sem_overall:.4f}")
print(f"Phase Cohesion (within-emotion): {cohesion_overall:.4f}")
print(f"Spread (S2 = 1 - max_cluster_share): {spread_overall:.4f}")
print(f"Cluster Purity: {purity:.4f}")
print(f"Neuro-Consistency (repeats): {neuro_overall:.4f}")
print(f"IA2 (phase alignment to REF-B1): {ia2_overall:.4f}")
print(f"Cognitive Integrity Index (CII): {CII:.4f}")
print("Report saved to:", os.path.abspath(REPORT_DIR))
print("Per-emotion metrics saved to:", os.path.join(REPORT_DIR, "per_emotion_metrics.json"))
print("Final metrics saved to:", os.path.join(REPORT_DIR, "final_metrics.json"))
print("Predictions saved to:", os.path.join(REPORT_DIR, "predictions.csv"))
print("UMAP/PCA visualization:", os.path.join(REPORT_DIR, "phase_field_umap.png"))
print("Confusion matrix:", os.path.join(REPORT_DIR, "confusion_matrix.png"))

# done
