import torch
import numpy as np
import inference
from phase_tokenizer import PhaseTokenizer

# Ensure tokenizer and a_temp initialized like in inference
if inference.tokenizer is None:
    inference.tokenizer = PhaseTokenizer(dim=128, h=0.05, i=1.0)
if inference.a_temp is None:
    inference.a_temp = torch.full((1, 1), 0.5, dtype=torch.float32)

print('[Inspect] building responder...')
responder = inference.build_chat_responder(max_examples=500)
print(f"[Inspect] entries={len(getattr(responder, 'entries', []))} centroids={len(getattr(responder, 'centroids', []))} anchors={len(getattr(responder, 'anchors', {}))}")

# Input to inspect
user_input = "describe your feelings"
print(f"[Inspect] user_input: {user_input}")

# Encode input and pass through model to get input phase
tokenizer = inference.tokenizer
phase_input = tokenizer.encode_text(user_input)
with torch.no_grad():
    _, input_phase = inference.model(phase_input, inference.a_temp, torch.tensor([0]), torch.tensor([0]), epoch=999)
if isinstance(input_phase, torch.Tensor):
    qp = input_phase.squeeze(0)
else:
    qp = torch.tensor(input_phase).squeeze(0)

print(f"[Inspect] input_phase mean={float(qp.mean()):.6f} std={float(qp.std()):.6f}")

# Check centroids
centroids = getattr(responder, 'centroids', [])
if centroids:
    cent_phases = torch.stack([c['phase'] for c in centroids], dim=0)
    sims = inference.phase_similarity_vector(qp, cent_phases).detach().cpu().numpy()
    order = np.argsort(-sims)
    print('\n[Inspect] Top 10 centroid matches:')
    for i in range(min(10, len(order))):
        idx = order[i]
        print(f"#{i+1}: sim={sims[idx]:.6f} count={centroids[idx].get('count', 1)} text='{centroids[idx]['text'][:120]}'")
else:
    print('[Inspect] no centroids')

# Check raw entries
entries = getattr(responder, 'entries', [])
if entries:
    cand_phases = torch.stack([e['phase'] for e in entries], dim=0)
    sims_e = inference.phase_similarity_vector(qp, cand_phases).detach().cpu().numpy()
    order_e = np.argsort(-sims_e)
    print('\n[Inspect] Top 20 candidate entry matches:')
    for i in range(min(20, len(order_e))):
        idx = order_e[i]
        text = inference._stringify_response(entries[idx].get('response'))
        print(f"#{i+1}: sim={sims_e[idx]:.6f} idx={idx} text='{text[:160]}'")

    # Check if any candidate looks like a logical reply (contains certain phrases)
    good_phrases = ["i am", "i'm", "i am fine", "i'm fine", "i'm good", "doing well", "i'm okay", "i am good", "i am fine,", "i'm great", "i'm doing fine", "i'm well"]
    found = []
    for i in range(min(200, len(order_e))):
        idx = order_e[i]
        text = inference._stringify_response(entries[idx].get('response')).lower()
        if any(p in text for p in good_phrases):
            found.append((i+1, idx, sims_e[idx], text[:160]))
    print('\n[Inspect] Logical-sounding replies among top-200:')
    if found:
        for rank, idx, sim, txt in found[:20]:
            print(f"rank={rank} idx={idx} sim={sim:.6f} text='{txt}'")
    else:
        print('None found')
else:
    print('[Inspect] no entries')

# Also check anchor matches
anchors = getattr(responder, 'anchors', {})
print(f"\n[Inspect] anchors size={len(anchors)}; sample tokens: {list(anchors.keys())[:10]}")

# Save diagnostics to file
np.savez('inspect_diag.npz', centroid_sims=sims if centroids else np.array([]), entry_sims=sims_e if entries else np.array([]))
print('[Inspect] saved inspect_diag.npz')
