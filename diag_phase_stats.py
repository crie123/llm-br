import torch
import numpy as np
import inference

responder = inference.build_chat_responder(max_examples=500)
entries = getattr(responder, 'entries', [])
print('entries_count=', len(entries))
if entries:
    for i in range(min(3, len(entries))):
        e = entries[i]
        print(f'entry[{i}] keys=', list(e.keys()))
        print(' response_snippet=', inference._stringify_response(e.get('response'))[:200])

    phases = torch.stack([e['phase'] for e in entries])
    print('phases shape=', phases.shape)
    print('phases mean=', float(phases.mean()), 'std=', float(phases.std()))

    norms = phases.norm(dim=1, keepdim=True)
    normalized = phases / (norms + 1e-12)
    sim_matrix = normalized @ normalized.t()
    sim_matrix = sim_matrix.cpu().numpy()

    # set diagonal to -1 to ignore self-similarity
    np.fill_diagonal(sim_matrix, -1.0)
    max_offdiag = sim_matrix.max(axis=1)
    print('max_offdiag: min=', float(max_offdiag.min()), 'mean=', float(max_offdiag.mean()), 'max=', float(max_offdiag.max()), 'std=', float(max_offdiag.std()))

    # overall distribution
    print('sim_matrix stats: min=', float(sim_matrix.min()), 'max=', float(sim_matrix.max()), 'mean=', float(sim_matrix.mean()))

    # save
    np.savez('diag_phase_stats.npz', phases=phases.cpu().numpy(), max_offdiag=max_offdiag, sim_matrix=sim_matrix)
    print('saved diag_phase_stats.npz')
else:
    print('no entries')
