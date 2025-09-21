import os
import sys
sys.path.append('project_libs')
import torch
import numpy as np
from phase_tokenizer import PhaseTokenizer
from neural_network import NeuralNetwork
from functions import selu
from sklearn.decomposition import PCA


def load_saved_model(path):
    if not os.path.exists(path):
        print(f"[ModelCheck] No model file at {path}")
        return None, None
    data = torch.load(path, map_location='cpu')
    state = None
    meta = {}
    if isinstance(data, dict) and 'model_state' in data:
        state = data['model_state']
        meta['input_size'] = data.get('input_size')
        meta['hidden_size'] = data.get('hidden_size')
        meta['output_size'] = data.get('output_size')
        meta['num_layers'] = data.get('num_layers')
        meta['phase_dim'] = data.get('phase_dim')
    else:
        # try using as state_dict directly
        state = data
    return state, meta


def weight_stats(model):
    stats = {}
    for name, p in model.named_parameters():
        arr = p.detach().cpu().numpy()
        mean = float(arr.mean())
        std = float(arr.std())
        zero_frac = float((arr == 0).sum()) / arr.size
        const = std == 0.0
        stats[name] = {'mean': mean, 'std': std, 'zero_frac': zero_frac, 'const': const, 'shape': arr.shape}
    return stats


def test_phase_tokenizer_variability(tokenizer):
    samples = [
        "",
        "a",
        "b",
        "hello",
        "hello!",
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog.",
        "1234567890",
        "!@#$%^&*()",
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor",
        "A slightly different sentence to observe embedding changes.",
        "A slightly different sentence to observe embedding changes!",
    ]
    embeds = []
    for s in samples:
        v = tokenizer.encode_text(s).squeeze(0).detach().cpu().numpy()
        embeds.append(v)
    E = np.stack(embeds)
    per_dim_std = E.std(axis=0)
    overall_std = E.std()
    pairwise_dists = np.sqrt(((E[:, None, :] - E[None, :, :]) ** 2).sum(axis=2))
    mean_pairwise = pairwise_dists.mean()
    print("[Tokenizer] Samples:")
    for i, s in enumerate(samples):
        print(f"  {i:2d}: '{s[:60]}' -> L2 from sample0: {np.linalg.norm(E[i]-E[0]):.6f}")
    print(f"[Tokenizer] Per-dimension std mean: {per_dim_std.mean():.6e}, overall std: {overall_std:.6e}")
    print(f"[Tokenizer] Mean pairwise L2 distance across {len(samples)} samples: {mean_pairwise:.6e}")
    return E


def compute_phase_pca(phases, n_components=10):
    # phases: (N, D)
    pca = PCA(n_components=min(n_components, phases.shape[1]))
    pca.fit(phases)
    explained = pca.explained_variance_ratio_
    print("[PCA] explained variance ratios (first 10):", explained[:10])
    return explained, pca


def reconstruct_pre_tanh(model, input_emb, emotion_id=0):
    # Re-run the NeuralNetwork forward loop to capture Z_total and pre-tanh (CPU)
    device = next(model.parameters()).device
    model_cpu = model.to('cpu')
    model_cpu.eval()
    with torch.no_grad():
        X = model_cpu.input_layer(input_emb.float())
        resonance = 0
        Z_total = None
        pre_tanh = None
        n_layers = len(model_cpu.wave_layers)
        for i, wave_layer in enumerate(model_cpu.wave_layers):
            X_out = wave_layer(X + resonance, y_true=torch.tensor([[emotion_id]], dtype=torch.float32), epoch=None)
            # compute func (selu) on detached numpy as in original code
            Z = torch.tensor(selu(X_out.detach().cpu().numpy()), dtype=torch.float32)
            if Z_total is None:
                Z_total = Z
            else:
                Z_total = Z_total + Z
            resonance = 0.8 * resonance + 0.2 * Z
            pre_tanh = Z_total / n_layers
            X = torch.tanh(pre_tanh)
        # final output layer
        out = model_cpu.output_layer(X, torch.tensor([[emotion_id]], dtype=torch.float32), epoch=None)
    return {
        'pre_tanh': pre_tanh.detach().cpu().numpy() if pre_tanh is not None else None,
        'Z_total': Z_total.detach().cpu().numpy() if Z_total is not None else None,
        'final_phase': X.detach().cpu().numpy(),
        'output_logits': out.detach().cpu().numpy()
    }


def main():
    print("[Diagnostics] Starting checks...")
    # load model if available
    state, meta = load_saved_model('pokrov_model.pt')
    model = None
    if state is not None:
        try:
            # use meta to build matching model
            if meta and meta.get('input_size') is not None:
                input_size = meta['input_size']
                hidden_size = meta['hidden_size'] or 32
                output_size = meta['output_size'] or 32
                num_layers = meta['num_layers'] or 4
                model = NeuralNetwork(input_size, hidden_size, output_size, num_layers=num_layers)
                model.load_state_dict(state)
                model.eval()
                print(f"[ModelCheck] Loaded model with input={input_size}, hidden={hidden_size}, out={output_size}, layers={num_layers}")
            else:
                print("[ModelCheck] No metadata found in model file; skipping model weight and phase PCA checks that require the model.")
                model = None
        except Exception as e:
            print(f"[ModelCheck] Failed to build/load model: {e}")
            model = None

    # Tokenizer tests
    # Prefer tokenizer dimensionality to match the model input when a model was loaded.
    if model is not None:
        tokenizer_dim = model.input_layer[0].in_features
    else:
        tokenizer_dim = meta.get('phase_dim') or 128
    tokenizer = PhaseTokenizer(dim=tokenizer_dim, h=0.05, i=1.0)
    E = test_phase_tokenizer_variability(tokenizer)

    # If model present, compute phases for many sample strings and run PCA
    if model is not None:
        # generate sample strings
        samples = []
        base = ["hello world", "goodbye", "I am happy", "I am sad", "angry", "excited", "The cat sat on the mat", "lorem ipsum dolor sit amet"]
        for i in range(200):
            samples.append(base[i % len(base)] + ("!" * (i % 5)) + (" extra" * (i % 7)))
        embeds = []
        for s in samples:
            v = tokenizer.encode_text(s).squeeze(0)
            embeds.append(v)
        X = torch.stack(embeds)
        # run model to get phases (second return)
        with torch.no_grad():
            logits, phase = model(X, torch.full((X.size(0), 1), 0.5), torch.zeros(X.size(0), dtype=torch.long), torch.zeros(X.size(0), dtype=torch.long), epoch=None)
        phases_np = phase.detach().cpu().numpy()
        print(f"[PhaseCollection] Collected phases shape: {phases_np.shape}")
        explained, pca = compute_phase_pca(phases_np, n_components=10)
        print(f"[PhasePCA] First component explains {explained[0]*100:.4f}% of variance")
        if explained[0] > 0.9:
            print("[PhasePCA] WARNING: first component explains >90% variance — possible collapse")

        # weight stats
        stats = weight_stats(model)
        print("[Weights] Parameter stats (mean/std/zero_frac/const) for top-level parameters:")
        for k, v in stats.items():
            print(f" {k}: mean={v['mean']:.6e}, std={v['std']:.6e}, zero_frac={v['zero_frac']:.3f}, const={v['const']}, shape={v['shape']}")

        # reconstruct pre-tanh for a few inputs
        for idx in [0, 5, 50, 199]:
            input_emb = tokenizer.encode_text(samples[idx])
            rec = reconstruct_pre_tanh(model, input_emb, emotion_id=0)
            pre = rec['pre_tanh']
            print(f"[PreTanh] sample {idx} pre_tanh mean={pre.mean():.6e}, std={pre.std():.6e}, shape={pre.shape}")
    else:
        print("[Diagnostics] Model not available — only tokenizer checks were run.")

    print("[Diagnostics] Done.")


if __name__ == '__main__':
    main()
