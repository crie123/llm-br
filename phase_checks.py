import torch
import numpy as np
import inference

# Ensure model and tokenizer initialized
if inference.tokenizer is None:
    from phase_tokenizer import PhaseTokenizer
    inference.tokenizer = PhaseTokenizer(dim=128, h=0.05, i=1.0)
if inference.a_temp is None:
    inference.a_temp = torch.full((1, 1), 0.5, dtype=torch.float32)

print('[PhaseChecks] building responder to load dataset and model...')
responder = inference.build_chat_responder(max_examples=500)
entries = getattr(responder, 'entries', [])
print('[PhaseChecks] entries=', len(entries))

if not entries:
    print('No entries found, aborting')
    raise SystemExit(1)

phases = torch.stack([e['phase'] for e in entries])
print('[PhaseChecks] phases shape=', phases.shape)

# L2 norms
norms = phases.norm(dim=1)
print('[PhaseChecks] norms: min=', float(norms.min()), 'mean=', float(norms.mean()), 'max=', float(norms.max()), 'std=', float(norms.std()))

# Per-dimension variance
per_dim_var = phases.var(dim=0, unbiased=False)
print('[PhaseChecks] per-dim var: min=', float(per_dim_var.min()), 'mean=', float(per_dim_var.mean()), 'max=', float(per_dim_var.max()))
small_var_count = (per_dim_var < 1e-8).sum().item()
print('[PhaseChecks] dims with var < 1e-8:', small_var_count, '/', phases.shape[1])

# Show top 10 dims by variance
order = torch.argsort(per_dim_var, descending=True)
print('[PhaseChecks] top dims by var (dim,var):')
for i in order[:10]:
    print(int(i), float(per_dim_var[i]))

# Test several diverse inputs through tokenizer+model
test_inputs = [
    "I am happy",
    "I am sad",
    "describe your feelings",
    "What's your name?",
    "The quick brown fox jumps over the lazy dog",
    "Hello",
    "Goodbye",
    "Tell me a joke",
    "I love you",
    "I'm hungry"
]

tokenizer = inference.tokenizer
model = inference.model
ate = inference.a_temp

test_phases = []
print('\n[PhaseChecks] Running test inputs through model...')
for t in test_inputs:
    enc = tokenizer.encode_text(t)
    with torch.no_grad():
        _, ph = model(enc, ate, torch.tensor([0]), torch.tensor([0]), epoch=999)
    if isinstance(ph, torch.Tensor):
        qp = ph.squeeze(0).cpu()
    else:
        qp = torch.tensor(ph).squeeze(0).cpu()
    test_phases.append(qp)
    print(f"input='{t}' norm={float(qp.norm()):.6f} mean={float(qp.mean()):.6f} std={float(qp.std()):.6f}")

# Pairwise similarities between test inputs
TP = torch.stack(test_phases)
# normalize then dot
normed = TP / (TP.norm(dim=1, keepdim=True) + 1e-12)
sim = (normed @ normed.t()).cpu().numpy()
print('\n[PhaseChecks] pairwise similarity matrix for test inputs:')
for i, row in enumerate(sim):
    print(i, ' '.join(f"{v:.6f}" for v in row))

# Save results
np.savez('phase_checks.npz', norms=norms.cpu().numpy(), per_dim_var=per_dim_var.cpu().numpy(), test_phases=TP.cpu().numpy(), sim=sim)
print('\n[PhaseChecks] saved phase_checks.npz')
