import os
import re
import torch
import numpy as np
import math
from typing import List, Dict, Any
from neural_network import NeuralNetwork
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from bitgrid import BitGridSensor, image_to_bitgrid, phase_to_bitgrid
import plotting as plotting

# Safe model load: don't fail import if checkpoint missing/corrupted
model = NeuralNetwork(input_size=128, hidden_size=32, output_size=32, num_layers=20)
ckpt_path = "pokrov_model.pt"
if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
    try:
        checkpoint = torch.load(ckpt_path)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            print(f"[Model] Loaded checkpoint from {ckpt_path}")
        else:
            print(f"[Model] checkpoint {ckpt_path} missing 'model_state' — using random init")
    except Exception as e:
        print(f"[Model] failed to load {ckpt_path}: {e} — continuing with random init")
else:
    print(f"[Model] checkpoint not found or empty ({ckpt_path}) — continuing with random init")

# Tokenizer and other heavy dataset preprocessing are created lazily at runtime (not on import)
tokenizer = None
bert_model = None
_label_encoder = None
a_temp = None

# PhaseComposer and helpers (for composing phase-based replies) -----------------
import math as _math

@torch.no_grad()
def topk_by_phase(query_phase: torch.Tensor, archive_phases: torch.Tensor, k: int = 5):
    """Return (indices, sims) of top-k by complex cosine similarity.
    query_phase: (dim,) in [-1,1]
    archive_phases: (N, dim)
    """
    q_phi = (query_phase.clamp(-1, 1) * _math.pi).to(archive_phases.device)
    a_phi = (archive_phases.clamp(-1, 1) * _math.pi).to(archive_phases.device)
    qz = torch.exp(1j * q_phi)
    az = torch.exp(1j * a_phi)
    sims = torch.real((qz * az.conj()).mean(dim=1))
    vals, idx = torch.topk(sims, k=min(k, sims.numel()))
    # ensure indices are CPU long tensor for safe Python use
    try:
        idx_cpu = idx.detach().to('cpu').long()
    except Exception:
        idx_cpu = idx.cpu().long()
    return idx_cpu, vals.detach().to('cpu')

# Phase-aware similarity helper (uses complex exponentials of phases)
@torch.no_grad()
def phase_similarity_vector(query_phase: torch.Tensor, archive_phases: torch.Tensor) -> torch.Tensor:
    """Compute similarity between query_phase and each archive_phase using complex phase inner product.
    Returns a tensor of shape (N,) on the archive_phases.device
    """
    try:
        q_phi = (query_phase.clamp(-1, 1) * _math.pi).to(archive_phases.device)
        a_phi = (archive_phases.clamp(-1, 1) * _math.pi).to(archive_phases.device)
        qz = torch.exp(1j * q_phi)
        az = torch.exp(1j * a_phi)
        sims = torch.real((qz * az.conj()).mean(dim=1))
        return sims
    except Exception:
        # fallback to cosine similarity
        return torch.nn.functional.cosine_similarity(query_phase.unsqueeze(0), archive_phases, dim=1).to(archive_phases.device)

class PhaseComposer:
    def __init__(self, dim: int = 128):
        self.dim = dim

    @staticmethod
    def softmax(x: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        x = x / max(t, 1e-6)
        x = x - x.max()
        return torch.exp(x) / (torch.exp(x).sum() + 1e-8)

    def compose(self, phases: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # More sophisticated phase mixing in complex domain
        phi = (phases.clamp(-1, 1) * _math.pi)
        z = torch.exp(1j * phi)
        w = weights.view(-1, 1)
        
        # Progressive mixing with stability check
        z_mix = torch.zeros_like(z[0], dtype=torch.complex64)
        stable_count = 0
        for i in range(z.size(0)):
            prev_z = z_mix.clone()
            z_mix = z_mix + w[i] * z[i]
            if i > 0:
                # Check if new contribution improves coherence
                delta = torch.abs(z_mix - prev_z).mean().item()
                if delta < 0.1:  # Small change threshold
                    stable_count += 1
                if stable_count >= 2:  # Stop if stable for a few steps
                    break
        
        # Normalize final result
        comp_phi = torch.angle(z_mix)
        comp_norm = torch.tensor(np.interp(comp_phi.cpu().numpy(), 
                                         (-_math.pi, _math.pi), 
                                         (-1, 1)), 
                               dtype=torch.float32, 
                               device=phases.device)
        return comp_norm

    def decode_to_text(self, phase: torch.Tensor, out_len: int = 64, guide_texts: List[str] = None) -> str:
        """Enhanced decoding that better preserves semantic meaning"""
        try:
            phi = (phase.clamp(-1, 1) * math.pi).cpu().numpy()
        except Exception:
            phi = (np.clip(np.array(phase), -1, 1) * math.pi)
        
        spec = np.exp(1j * phi)
        seq = np.fft.ifft(spec).real
        
        # Improved normalization with smoothing
        window = min(5, len(seq) // 10)
        smoothed = np.convolve(seq, np.ones(window)/window, mode='valid')
        rng = smoothed.max() - smoothed.min()
        if rng <= 1e-6:
            norm = np.zeros_like(smoothed)
        else:
            norm = (smoothed - smoothed.min()) / rng

        # Enhanced charset with more natural language tokens
        charset = list("abcdefghijklmnopqrstuvwxyz0123456789 ,.!?')(\"+-*/=<>[]{}@#$%^&")
        L = len(charset)
        indices = (norm * (L - 1)).astype(int)[:out_len]
        chars = [charset[i] for i in indices]
        s = ''.join(chars)
        
        # Better text cleanup
        s = re.sub(r"(\s){2,}", " ", s)
        s = re.sub(r"([^\w\s])\1{2,}", r"\1", s)  # Collapse repeated punctuation
        s = s.strip(" '\"")
        
        # Ensure sentence structure
        if s and not any(s.endswith(p) for p in '.!?'):
            s = s + '.'
            
        # Prefer longer guide texts when they seem meaningful
        if guide_texts:
            candidates = [t.strip() for t in guide_texts if t and isinstance(t, str)]
            if candidates:
                # Score by length and presence of complete sentences
                scores = [(i, len(c.split()), sum(1 for x in c if x in '.!?')) 
                         for i, c in enumerate(candidates)]
                best_idx = max(scores, key=lambda x: (x[2], x[1]))[0]
                best = candidates[best_idx]
                
                if len(best.split()) > max(15, out_len // 4):
                    return best
                
                # Combine best guide with decoded for better coherence
                merged = (best + ' — ' + s).strip()
                return ' '.join(merged.split())

        return ' '.join(s.split())

    def compose_text(self,
                     query_phase: torch.Tensor,
                     cand_phases: torch.Tensor,
                     cand_texts: List[str],
                     temperature: float = 0.5,
                     k: int = 5,
                     out_len: int = 160) -> Dict[str, Any]:
        idx, sims = topk_by_phase(query_phase, cand_phases, k=k)
        # idx and sims are ensured to be on CPU by topk_by_phase
        # clamp indices to valid range as a safety measure
        try:
            max_idx = max(0, cand_phases.size(0) - 1)
            idx = idx.clamp(0, max_idx)
        except Exception:
            pass

        w = self.softmax(sims, t=temperature)
        # use indices to select phases
        try:
            mixed = self.compose(cand_phases[idx], w)
        except Exception:
            # fallback: use top k from start
            mixed = self.compose(cand_phases[:k], self.softmax(sims[:k], t=temperature))

        # pass picked texts as guidance to the decoder for more coherent speech
        try:
            picked = [cand_texts[int(i)] for i in idx.tolist()] if cand_texts else []
        except Exception:
            picked = []
        text = self.decode_to_text(mixed, out_len=out_len, guide_texts=picked)
        return {
            'indices': idx.tolist(),
            'sims': sims.tolist(),
            'weights': w.tolist(),
            'picked_texts': picked,
            'composed_text': text,
            'mixed_phase': mixed.detach().cpu() if isinstance(mixed, torch.Tensor) else torch.tensor(mixed)
        }


# Simple bit-grid utilities (useful for training/visualization modules)
def iou_bits(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.bool_)
    b = b.astype(np.bool_)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


# Helper: flatten stored dataset 'response' into readable string
def _stringify_response(resp) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        parts = []
        for turn in resp:
            if isinstance(turn, dict):
                if turn.get('role') == 'assistant' and 'content' in turn:
                    parts.append(str(turn['content']))
                elif 'content' in turn:
                    parts.append(str(turn['content']))
                elif 'text' in turn:
                    parts.append(str(turn['text']))
                else:
                    parts.append(' '.join(str(v) for v in turn.values()))
            else:
                parts.append(str(turn))
        return ' '.join(p for p in parts if p)
    if isinstance(resp, dict):
        return str(resp.get('content') or resp.get('text') or resp)
    return str(resp)


# Compose a reply from top-K dataset phases using PhaseComposer
composer = PhaseComposer(dim=128)

@torch.no_grad()
def mc_consensus_pick(query_phase: torch.Tensor, cand_phases: torch.Tensor, n_iters: int = 12, sigma: float = 0.015, vote_threshold: float = 0.5):
    """Perturb query_phase n_iters times and collect top-1 picks; return index if a candidate wins >= vote_threshold fraction."""
    try:
        counts = {}
        N = cand_phases.size(0)
        for _ in range(n_iters):
            noise = torch.randn_like(query_phase) * sigma
            qp_pert = (query_phase + noise).clamp(-1.0, 1.0)
            sims = phase_similarity_vector(qp_pert, cand_phases).detach().cpu()
            top_idx = int(torch.argmax(sims).item())
            counts[top_idx] = counts.get(top_idx, 0) + 1
        # find winner
        winner, wins = max(counts.items(), key=lambda kv: kv[1]) if counts else (None, 0)
        if winner is not None and wins >= max(1, int(n_iters * vote_threshold)):
            return int(winner), wins / float(n_iters)
    except Exception as e:
        print(f"[MC] failed: {e}")
    return None, 0.0


def heuristic_rerank_for_simple_q(user_text: str, cand_texts: List[str], cand_sims: torch.Tensor, top_k: int = 50, sim_margin: float = 0.03):
    """If user asks a common question type, try to find a logical reply among top_k candidates.
    Return index if found and sim is within sim_margin of best_sim.
    """
    if not user_text or not isinstance(user_text, str):
        return None
    ut = user_text.strip().lower()
    
    # Detect question types
    is_greeting = re.match(r'^(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings?)\b', ut)
    is_how_are_you = re.search(r"\b(how\s+are\s+you|how\'?s?\s+it\s+going|how\s+do\s+you\s+do)\b", ut)
    is_who_what = re.match(r'^(who|what)\s+are\s+you\b', ut)
    is_name_q = re.search(r"\b(what(?:'s|\s+is)\s+your\s+name)\b", ut)
    
    # Skip if not a recognized pattern
    if not (is_greeting or is_how_are_you or is_who_what or is_name_q):
        return None
        
    try:
        sims_cpu = cand_sims.detach().cpu()
        k = min(top_k, sims_cpu.numel())
        vals, idxs = torch.topk(sims_cpu, k=k)
        best_sim = float(vals[0].item()) if vals.numel() > 0 else -1.0
        
        good_patterns = {
            'greeting': ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"],
            'how_are_you': ["i'm", "i am", "i'm fine", "i am fine", "i'm good", "i am good", "i'm okay", "i'm well", "i'm great"],
            'who_what': ["i'm", "i am", "my name is", "i'm an ai", "i am an ai", "i'm your", "i am your"],
            'name': ["my name", "i'm called", "i am called", "you can call me"]
        }

        # Pick appropriate patterns based on question type
        if is_greeting:
            patterns = good_patterns['greeting']
        elif is_how_are_you:
            patterns = good_patterns['how_are_you']
        elif is_who_what:
            patterns = good_patterns['who_what']
        elif is_name_q:
            patterns = good_patterns['name']
        
        for i in range(idxs.numel()):
            ci = int(idxs[i].item())
            txt = (cand_texts[ci] or "").lower()
            # For each pattern category, check if the response contains any good patterns
            if any(p in txt for p in patterns):
                sim_val = float(vals[i].item())
                if sim_val >= best_sim - sim_margin:
                    return ci
                    
    except Exception as e:
        print(f"[HEUR] failed: {e}")
    return None

def select_response(query_phase: torch.Tensor, cand_phases: torch.Tensor, cand_texts: List[str], user_text: str) -> Dict[str, Any]:
    """Enhanced response selection that considers semantic and lexical matching"""
    # Get initial similarity scores
    sims = phase_similarity_vector(query_phase, cand_phases)
    
    # Boost scores based on word overlap
    if user_text:
        user_words = set(w.lower() for w in re.findall(r'\w+', user_text))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        user_words = user_words - stop_words
        
        # Calculate word overlap scores
        overlap_scores = torch.zeros_like(sims)
        for i, text in enumerate(cand_texts):
            if not text:
                continue
            resp_words = set(w.lower() for w in re.findall(r'\w+', text))
            overlap = len(user_words & resp_words)
            if overlap > 0:
                # Scale overlap boost by response length
                scale = min(1.0, 5.0 / max(1, len(resp_words)))
                overlap_scores[i] = overlap * scale * 0.1  # Small boost per word

        # Apply overlap boost
        sims = sims + overlap_scores

    # Get top candidates
    vals, idxs = torch.topk(sims, k=min(5, len(cand_texts)))
    
    best_idx = idxs[0]
    best_sim = vals[0]
    
    # If best match is very strong, return it directly
    if best_sim > 0.85:
        return {
            'index': int(best_idx),
            'sim': float(best_sim),
            'is_direct': True
        }
        
    # Otherwise check coherence among top matches
    coherent_idxs = []
    for i, idx in enumerate(idxs):
        text = cand_texts[int(idx)]
        if not text:
            continue
            
        # Check if response is well-formed
        if len(text.split()) >= 3 and any(text.endswith(p) for p in '.!?'):
            coherent_idxs.append((i, int(idx), float(vals[i])))
            
    if coherent_idxs:
        # Return highest scoring coherent response
        _, idx, sim = coherent_idxs[0]
        return {
            'index': idx,
            'sim': sim,
            'is_direct': False
        }
        
    # Fallback to best match
    return {
        'index': int(best_idx),
        'sim': float(best_sim),
        'is_direct': False
    }

def compose_reply_from_responder(query_phase: torch.Tensor, responder_obj, k: int = 5, temperature: float = 0.6, out_len: int = 100, use_l2_norm: bool = False, use_phase_sim: bool = True, user_text: str = None) -> Dict[str, Any]:
    if not getattr(responder_obj, 'entries', None):
        return {'composed_text': '', 'picked_texts': [], 'indices': [], 'sims': [], 'weights': []}

    cand_phases = torch.stack([e['phase'] for e in responder_obj.entries], dim=0)
    cand_texts = [_stringify_response(e.get('response')) for e in responder_obj.entries]

    # If user_text provided, prepare token set for anchor matching
    anchors_tokens = None
    if user_text and isinstance(user_text, str):
        try:
            toks = re.findall(r"\w+", user_text.lower())
            stop = set(['the','and','a','an','to','of','in','on','is','it','i','you','are','for','with','that','this','was','were','be'])
            anchors_tokens = [t for t in toks if len(t) > 2 and t not in stop]
        except Exception:
            anchors_tokens = None

    # Check for centroid consensus first
    if hasattr(responder_obj, 'centroids') and responder_obj.centroids:
        try:
            centroid_phases = torch.stack([c['phase'] for c in responder_obj.centroids], dim=0)
            centroid_texts = [c['text'] for c in responder_obj.centroids]
            centroid_sims = phase_similarity_vector(query_phase, centroid_phases)
            best_idx = int(torch.argmax(centroid_sims).item())
            best_sim = float(centroid_sims[best_idx].item())
            SIM_THRESHOLD = 0.85
            if best_sim >= SIM_THRESHOLD:
                return {
                    'composed_text': centroid_texts[best_idx],
                    'picked_texts': [centroid_texts[best_idx]],
                    'indices': [best_idx],
                    'sims': [best_sim],
                    'weights': [1.0],
                    'mixed_phase': centroid_phases[best_idx]
                }
            else:
                # If no direct centroid match, but the best centroid is reasonably close,
                # nudge the query phase towards that centroid in complex phase domain.
                try:
                    if best_sim > 0.7:
                        alpha = 0.6  # how strongly to nudge towards centroid
                        # convert phases to complex exponentials
                        qp_phi = (query_phase.clamp(-1,1) * _math.pi).to(centroid_phases.device)
                        cent_phi = (centroid_phases[best_idx].clamp(-1,1) * _math.pi).to(centroid_phases.device)
                        qz = torch.exp(1j * qp_phi)
                        cz = torch.exp(1j * cent_phi)
                        z_mix = (1.0 - alpha) * qz + alpha * cz
                        mixed_phi = torch.angle(z_mix) / _math.pi
                        # mixed_phi in [-1,1]
                        query_phase = mixed_phi.to(query_phase.device)
                        print(f"[DBG_NUDGE] nudged query_phase toward centroid idx={best_idx} alpha={alpha} best_sim={best_sim:.6f}")
                except Exception as _e:
                    print(f"[DBG_NUDGE] failed to nudge toward centroid: {_e}")
        except Exception as e:
            print(f"[DBG_CENTROID] failed to use centroids: {e}")

    # Anchor boosting: if responder has anchors mapping words->centroid indices, boost sims accordingly
    anchor_boosts = None
    if anchors_tokens and hasattr(responder_obj, 'anchors'):
        try:
            anchor_boosts = np.zeros(cand_phases.size(0), dtype=np.float32)
            for t in anchors_tokens:
                idxs = responder_obj.anchors.get(t)
                if not idxs:
                    continue
                for ci in idxs:
                    if 0 <= ci < anchor_boosts.size:
                        anchor_boosts[ci] += 1.0
            if anchor_boosts.sum() > 0:
                # normalize and scale boost with clipping
                anchor_boosts = anchor_boosts / (anchor_boosts.max() + 1e-9) * 0.02
                anchor_boosts = np.clip(anchor_boosts, 0.0, 0.05)  # Limit boost/dampening effect
                anchor_boosts = torch.tensor(anchor_boosts, dtype=torch.float32, device=cand_phases.device)
                print(f"[DBG_ANCHOR] tokens={anchors_tokens} applied anchor boosts")
        except Exception as _e:
            print(f"[DBG_ANCHOR] failed to compute anchor boosts: {_e}")

    # compute similarity: prefer phase-aware complex similarity
    qp = query_phase.squeeze(0) if query_phase.dim() == 2 else query_phase
    if use_phase_sim:
        sims = phase_similarity_vector(qp, cand_phases)
    else:
        # optionally L2-normalize vectors before computing similarity if requested
        if use_l2_norm:
            try:
                qp_norm = qp / (qp.norm(p=2) + 1e-9)
                cand_norm = cand_phases / (cand_phases.norm(p=2, dim=1, keepdim=True) + 1e-9)
                sims = torch.matmul(cand_norm, qp_norm.unsqueeze(-1)).squeeze(-1).cpu()
                sims = sims.to(cand_phases.device)
                print(f"[DBG_NORM] L2 normalization enabled for similarity computation")
            except Exception as _e:
                print(f"[DBG_NORM] failed to normalize: {_e}")
                sims = torch.nn.functional.cosine_similarity(qp.unsqueeze(0), cand_phases, dim=1)
        else:
            sims = torch.nn.functional.cosine_similarity(qp.unsqueeze(0), cand_phases, dim=1)

    # apply anchor boosts if present
    try:
        if anchor_boosts is not None:
            sims = sims + anchor_boosts
    except Exception:
        pass

    # Heuristic rerank for simple social/greeting questions — quick win
    try:
        hr_idx = heuristic_rerank_for_simple_q(user_text, cand_texts, sims, top_k=80, sim_margin=0.035)
        if hr_idx is not None:
            print(f"[HEUR] heuristic rerank selected idx={hr_idx}")
            chosen = cand_texts[hr_idx]
            trimmed = trim_reply(chosen, max_sentences=2, max_words=60)
            return {
                'composed_text': trimmed,
                'picked_texts': [chosen],
                'indices': [hr_idx],
                'sims': [float(sims[hr_idx].item())],
                'weights': [1.0],
                'mixed_phase': cand_phases[hr_idx]
            }
    except Exception:
        pass

    # Monte-Carlo consensus voting to 'stabilize' choice in dense clusters
    try:
        mc_idx, mc_conf = mc_consensus_pick(qp, cand_phases, n_iters=16, sigma=0.02, vote_threshold=0.55)
        if mc_idx is not None:
            print(f"[MC] consensus pick idx={mc_idx} conf={mc_conf:.2f}")
            chosen = cand_texts[mc_idx]
            trimmed = trim_reply(chosen, max_sentences=3, max_words=80)
            return {
                'composed_text': trimmed,
                'picked_texts': [chosen],
                'indices': [mc_idx],
                'sims': [float(sims[mc_idx].item())],
                'weights': [1.0],
                'mixed_phase': cand_phases[mc_idx]
            }
    except Exception:
        pass

    best_idx = int(torch.argmax(sims).item())
    best_sim = float(sims[best_idx].item())

    # Decision logic: return dataset candidate only if it's clearly the best match
    # Require absolute similarity and a margin over the second-best to avoid returning
    # dataset replies for weak/ambiguous matches.
    # tighten thresholds since clusters are dense
    SIM_THRESHOLD = 0.80  # Lowered from 0.90
    MARGIN = 0.02  # Lowered from 0.03 to allow more matches
    # compute second-best
    try:
        sorted_vals, _ = torch.sort(sims.detach().cpu(), descending=True)
        second_best = float(sorted_vals[1].item()) if sorted_vals.numel() > 1 else -1.0
    except Exception:
        second_best = -1.0

    def trim_reply(text, max_sentences=3, max_words=60):
        if not text:
            return ''
        import re
        # normalize separators used in dataset (many use multiple spaces or newlines)
        text = re.sub(r"\s{2,}", " ", text.replace('\n', ' ').strip())
        # split on sentence-ending punctuation
        parts = re.split(r'(?<=[.!?])\s+', text)
        parts = [p.strip() for p in parts if p.strip()]
        out = ' '.join(parts[:max_sentences]) if parts else text
        # ensure not too long in words
        words = out.split()
        if (len(words) > max_words):
            out = ' '.join(words[:max_words]).rstrip(' ,;:') + '...'
        return out

    take_direct = False
    if best_sim >= SIM_THRESHOLD and (best_sim - second_best) >= MARGIN:
        take_direct = True
    try:
        print(f"[DBG_DECISION] best={best_sim:.6f} second={second_best:.6f} margin={best_sim-second_best:.6f} take_direct={take_direct}")
    except Exception:
        pass

    if take_direct:
        # Return the dataset candidate directly (trimmed).
        try:
            print(f"[DBG_PATH] best_sim={best_sim:.6f} >= {SIM_THRESHOLD} and margin ok; returning dataset candidate idx={best_idx}")
        except Exception:
            pass
        chosen = cand_texts[best_idx]
        trimmed = trim_reply(chosen, max_sentences=3, max_words=60)
        return {
            'composed_text': trimmed,
            'picked_texts': [cand_texts[best_idx]],
            'indices': [best_idx],
            'sims': [best_sim],
            'weights': [1.0],
            'mixed_phase': cand_phases[best_idx]
        }

    # if anchor_boosts pointed to a centroid with decent boosted sim, consider direct return
    try:
        if anchor_boosts is not None and anchor_boosts.sum() > 0:
            boosted_idx = int(torch.argmax(sims).item())
            boosted_sim = float(sims[boosted_idx].item())
            if boosted_sim >= 0.80:
                print(f"[DBG_ANCHOR_PATH] returning centroid due to anchors idx={boosted_idx} sim={boosted_sim:.6f}")
                chosen = cand_texts[boosted_idx]
                trimmed = trim_reply(chosen, max_sentences=3, max_words=60)
                return {
                    'composed_text': trimmed,
                    'picked_texts': [cand_texts[boosted_idx]],
                    'indices': [boosted_idx],
                    'sims': [boosted_sim],
                    'weights': [1.0],
                    'mixed_phase': cand_phases[boosted_idx]
                }
    except Exception:
        pass

    # Otherwise, compose a fresh phase by mixing the top-k phases using phase-aware similarity
    try:
        topk = min(max(3, k), cand_phases.size(0))
        idxs, top_vals = topk_by_phase(qp, cand_phases, k=topk)
        # topk_by_phase returns CPU indices and vals; build weights and move to device
        try:
            vals_tensor = torch.tensor(top_vals, dtype=torch.float32)
            weights = composer.softmax(vals_tensor, t=temperature).to(cand_phases.device)
        except Exception:
            vals_tensor = torch.tensor(top_vals, dtype=torch.float32)
            weights = torch.softmax(vals_tensor / max(temperature, 1e-6), dim=0).to(cand_phases.device)
        idxs_device = idxs.to(cand_phases.device)
        mixed = composer.compose(cand_phases[idxs_device], weights)
        # decode with the top-k texts as guidance (prefer longer informative guides)
        guide_texts = [cand_texts[int(i)] for i in idxs.tolist() if int(i) < len(cand_texts)]
        # sort guide_texts by length descending so decoder prefers informative guidance
        guide_texts = sorted(guide_texts, key=lambda s: -len(s.split()))
        decoded = composer.decode_to_text(mixed, out_len=out_len, guide_texts=guide_texts)
        # if decoded sketch is too short or looks like gibberish, prefer best guide text trimmed
        if len(decoded.strip()) < 8 and guide_texts:
            decoded = trim_reply(guide_texts[0], max_sentences=2, max_words=120)
        return {
            'composed_text': decoded,
            'picked_texts': guide_texts,
            'indices': idxs.tolist(),
            'sims': top_vals.tolist() if hasattr(top_vals, 'tolist') else list(top_vals),
            'weights': weights.cpu().tolist() if hasattr(weights, 'cpu') else list(weights),
            'mixed_phase': mixed.detach().cpu() if isinstance(mixed, torch.Tensor) else torch.tensor(mixed)
        }
    except Exception as e:
        print(f"[DBG_COMPOSE] composition failed: {e} — falling back to previous composer path")
        top_candidate = cand_texts[best_idx]
        top_trim = trim_reply(top_candidate, max_sentences=3, max_words=60)
        guide = [top_trim] if top_trim else []
        comp = composer.compose_text(qp, cand_phases, guide, temperature=temperature, k=min(k, cand_phases.size(0)), out_len=out_len)
        comp_text = trim_reply(comp.get('composed_text', ''), max_sentences=3)
        comp['composed_text'] = comp_text
        return comp

def build_chat_responder(max_examples: int = 10000):
    from neural_network import EmpathicDatasetResponder
    global tokenizer, a_temp, _label_encoder
    
    if tokenizer is None:
        from phase_tokenizer import PhaseTokenizer
        tokenizer = PhaseTokenizer(dim=128, h=0.05, i=1.0)
    if a_temp is None:
        a_temp = torch.full((1, 1), 0.5, dtype=torch.float32)

    # First load all available Hugging Face datasets
    datasets_to_load = [
        ("daily_dialog", None),
        ("empathetic_dialogues", None),
        ("IlyaGusev/ruPersonaChat", None),
        ("SiberiaSoft/SiberianPersonaChat", None),
    ]
    
    all_examples = []
    
    for dataset_name, subset_size in datasets_to_load:
        try:
            print(f"[ChatResponder] Loading {dataset_name}...")
            try:
                ds = load_dataset(dataset_name, split='train', trust_remote_code=True)
            except Exception:
                try:
                    ds = load_dataset(dataset_name, trust_remote_code=True)
                except Exception as e:
                    print(f"[ChatResponder] Failed to load {dataset_name}: {e}")
                    continue
            
            if ds is not None:
                if subset_size:
                    ds = ds.select(range(min(len(ds), subset_size)))
                else:
                    ds = ds.select(range(len(ds)))
                    
                if dataset_name == "SiberiaSoft/SiberianPersonaChat":
                    ru_examples = []
                    for d in ds:
                        if "input" in d and "output" in d:
                            input_lines = d["input"].split("\n")
                            for line in input_lines:
                                if line.startswith("Собеседник:"):
                                    last_user_msg = line.replace("Собеседник:", "").strip()
                            
                            if last_user_msg and d["output"].strip():
                                ru_examples.append({
                                    "context": last_user_msg,
                                    "response": d["output"].strip(),
                                    "emotion": "neutral"
                                })
                    all_examples.extend(ru_examples)
                    print(f"[ChatResponder] Added {len(ru_examples)} Russian examples from SiberianPersonaChat")
                else:
                    all_examples.extend(ds)
                    print(f"[ChatResponder] Added {len(ds)} examples from {dataset_name}")
        except Exception as e:
            print(f"[ChatResponder] Failed to load {dataset_name}: {e}")
            continue

    # Load CIFAR-10 images and convert to phase embeddings
    try:
        print("[ChatResponder] Loading CIFAR-10 images...")
        ds_imgs = load_dataset("cifar10", split="train[:1000]")  # Take first 1000 images
        sensor = BitGridSensor(dim=128, grid=16, threshold=0.5)
        cifar_examples = []
        
        for item in ds_imgs:
            try:
                img = np.asarray(item['img'].convert('L').resize((128, 128)), dtype=np.float32) / 255.0
                phase = sensor.encode_image(img)
                if isinstance(phase, torch.Tensor):
                    phase = phase.detach().cpu()
                
                # Create an example with the image phase
                cifar_examples.append({
                    "phase": phase,
                    "response": f"This is image #{item['label']} from the CIFAR-10 dataset",
                    "emotion": "neutral"
                })
            except Exception as e:
                print(f"[ChatResponder] Failed to process CIFAR image: {e}")
                continue
                
        if cifar_examples:
            all_examples.extend(cifar_examples)
            print(f"[ChatResponder] Added {len(cifar_examples)} CIFAR-10 image examples")
    except Exception as e:
        print(f"[ChatResponder] Failed to load CIFAR-10: {e}")

    # Try loading Cornell Movie Dialog corpus from local files if available
    try:
        import os
        import csv
        cornell_path = os.path.join("datasets", "cornell_movie_dialogs")
        if os.path.exists(cornell_path):
            print("[ChatResponder] Loading Cornell Movie Dialogs from local files...")
            
            # Load movie conversations
            conversations_file = os.path.join(cornell_path, "movie_conversations.txt")
            lines_file = os.path.join(cornell_path, "movie_lines.txt")
            
            # First load all lines
            movie_lines = {}
            try:
                with open(lines_file, 'r', encoding='iso-8859-1') as f:
                    for line in f:
                        parts = line.strip().split(" +++$+++ ")
                        if len(parts) >= 5:
                            line_id = parts[0]
                            text = parts[4]
                            movie_lines[line_id] = text
            except Exception as e:
                print(f"[ChatResponder] Failed to load movie lines: {e}")
            
            # Then process conversations
            try:
                cornell_examples = []
                with open(conversations_file, 'r', encoding='iso-8859-1') as f:
                    for line in f:
                        parts = line.strip().split(" +++$+++ ")
                        if len(parts) >= 4:
                            # Extract line IDs from conversation
                            line_ids = eval(parts[3])  # Convert string repr of list to actual list
                            for i in range(len(line_ids)-1):
                                context = movie_lines.get(line_ids[i], "")
                                response = movie_lines.get(line_ids[i+1], "")
                                if context and response:
                                    cornell_examples.append({
                                        "context": context,
                                        "response": response,
                                        "emotion": "neutral"  # Cornell doesn't have emotion labels
                                    })
                if cornell_examples:
                    # Take up to max_examples/3 from Cornell to balance with other datasets
                    subset_size = max(1000, max_examples // 3)
                    cornell_subset = cornell_examples[:subset_size]
                    all_examples.extend(cornell_subset)
                    print(f"[ChatResponder] Added {len(cornell_subset)} examples from Cornell Movie Dialogs")
            except Exception as e:
                print(f"[ChatResponder] Failed to process movie conversations: {e}")
    except Exception as e:
        print(f"[ChatResponder] Failed to load Cornell dataset: {e}")

    if not all_examples:
        print("[ChatResponder] Warning: No datasets loaded - proceeding with empty dataset")
        all_examples = []
    else:
        print(f"[ChatResponder] Total examples loaded: {len(all_examples)}")
    
    # Prepare label encoder with all possible emotions
    emotions = set()
    for ex in all_examples:
        if isinstance(ex, dict):
            emo = ex.get("emotion") or ex.get("label")
            if emo:
                emotions.add(str(emo).lower())
    
    if not emotions:
        emotions = {"neutral"}
    
    if _label_encoder is None:
        _label_encoder = LabelEncoder()
    _label_encoder.fit(list(emotions))
    
    # Create responder with all examples
    responder = EmpathicDatasetResponder(
        dataset=all_examples,
        label_encoder=_label_encoder,
        bert_model=None,
        tokenizer=tokenizer,
        model=model,
        a_temp=a_temp
    )

    # Build centroids and anchors
    if hasattr(responder, 'entries') and responder.entries:
        try:
            # Build centroid map
            text_bins = {}
            for e in responder.entries:
                txt = _stringify_response(e.get('response'))
                if not txt:
                    continue
                if txt not in text_bins:
                    text_bins[txt] = []
                ph = e.get('phase')
                if isinstance(ph, torch.Tensor):
                    text_bins[txt].append(ph.detach().cpu().float())
                else:
                    text_bins[txt].append(torch.tensor(ph, dtype=torch.float32))

            centroids = []
            for txt, ph_list in text_bins.items():
                try:
                    stack = torch.stack(ph_list, dim=0)
                    phi = (stack.clamp(-1,1) * math.pi).numpy()
                    z = np.exp(1j * phi)
                    z_mean = z.mean(axis=0)
                    ang = np.angle(z_mean)
                    cent_phase = np.interp(ang, (-math.pi, math.pi), (-1, 1)).astype(np.float32)
                    cent_t = torch.tensor(cent_phase, dtype=torch.float32)
                    centroids.append({'text': txt, 'phase': cent_t, 'count': len(ph_list)})
                except Exception:
                    continue

            responder.centroids = centroids
            print(f"[ChatResponder] Built {len(centroids)} centroids from {len(responder.entries)} entries")

            # Build anchor map with less aggressive thresholds
            try:
                anchors = {}
                for idx, c in enumerate(centroids):
                    toks = re.findall(r"\w+", c['text'].lower())
                    for t in toks:
                        if len(t) <= 2:  # Keep short words that might be important (e.g. "hi", "ok")
                            continue
                        anchors.setdefault(t, []).append(idx)
                responder.anchors = anchors
                print(f"[ChatResponder] Built anchors for {len(anchors)} tokens")
            except Exception as e:
                print(f"[ChatResponder] Failed to build anchors: {e}")

        except Exception as e:
            print(f"[ChatResponder] Failed to build centroids: {e}")

    return responder

def evaluate_bitgrid_recognition(responder_obj, num_images: int = 200, grid: int = 16, subset: str = "train[:%d]" % 200):
    """Load CIFAR10 subset, for each image compute ground-truth bitgrid and query composer with image phase.
    Measure IoU and bit accuracy between composed phase->bitgrid and true image bitgrid. Plot distributions.
    """
    try:
        # BitGridSensor imported from bitgrid above
        sensor = BitGridSensor(dim=128, grid=16, threshold=0.5)
    except Exception:
        print("[Eval] bitgrid module missing")
        return {}

    try:
        ds_imgs = load_dataset("cifar10", split=f"train[:{num_images}]")
    except Exception as e:
        print(f"[Eval] can't load cifar10: {e} — skipping evaluation")
        return {}

    ious = []
    accs = []
    for item in ds_imgs:
        img = np.asarray(item['img'].convert('L').resize((128,128)), dtype=np.float32) / 255.0
        gt_bits = image_to_bitgrid(img, g=grid)
        query_phase = sensor.encode_image(img)
        # ensure tensor shape
        if isinstance(query_phase, torch.Tensor):
            qp = query_phase.unsqueeze(0)
        else:
            qp = torch.tensor(query_phase).unsqueeze(0)
        res = compose_reply_from_responder(qp.squeeze(0), responder_obj, k=6, temperature=0.6, out_len=120)
        mixed_phase = res.get('mixed_phase')
        if isinstance(mixed_phase, torch.Tensor):
            mixed_phase = mixed_phase.squeeze(0).cpu()
        else:
            mixed_phase = torch.tensor(mixed_phase)
        # use phase_to_bitgrid implemented here (keeps conversion local)
        pred_bits = phase_to_bitgrid(mixed_phase, g=grid)
        ious.append(iou_bits(gt_bits, pred_bits))
        accs.append(float((gt_bits == pred_bits).mean()))

    stats = {
        'mean_iou': float(np.mean(ious)) if ious else 0.0,
        'mean_bit_acc': float(np.mean(accs)) if accs else 0.0,
        'samples': len(ious),
        'ious': ious,
        'accs': accs
    }
    print(f"[Eval] Bitgrid recognition over {stats['samples']} samples: mIoU={stats['mean_iou']:.4f}, bit_acc={stats['mean_bit_acc']:.4f}")

    # Plot metrics using plotting helper
    try:
        plotting.plot_bitgrid_metrics(ious, accs)
    except Exception as e:
        print(f"[Eval] plotting failed: {e}")

    return stats

def print_model_info():
    """Print simple model load diagnostics: number of parameters and whether any non-default state found."""
    try:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[ModelInfo] total_params={n_params}, trainable_params={n_trainable}")
        # check for non-zero parameters as heuristic for loaded checkpoint
        any_nonzero = any((p.data.abs().sum().item() > 1e-6) for p in model.parameters())
        print(f"[ModelInfo] any_nonzero_params={any_nonzero}")
    except Exception as e:
        print(f"[ModelInfo] failed to inspect model: e")

def make_direct_reply(user_input: str, reply_text: str) -> str:
    """Produce a concise, directly relevant reply for the user.
    Heuristics:
    - If user asked a question, ensure response answers directly
    - If user greeted or made small talk, keep response brief and friendly
    - Otherwise, return a concise response focused on main point
    """
    import re
    if not reply_text or not isinstance(reply_text, str):
        return "Sorry, I don't have an answer right now."

    ui = (user_input or "").strip().lower()
    rt = reply_text.strip()
    
    # Special handling for common interactions
    if re.match(r'^(hi|hello|hey|привет|здравствуй|здравствуйте)\b', ui):
        return "Hi! How can I help you?"
        
    if re.search(r"\b(how\s+are\s+you|how\'?s?\s+it\s+going)\b", ui):
        return "I'm doing well, thank you! How can I assist you?"
        
    if re.match(r'^(who|what)\s+are\s+you\b', ui):
        return "I'm an AI assistant here to help you. What would you like to know?"
        
    if re.search(r"\b(what(?:'s|\s+is)\s+your\s+name)\b", ui):
        return "I'm Architect. How can I help you today?"

    # For other questions, ensure direct answers
    interrogatives = ('who', 'what', 'when', 'where', 'why', 'how', 'do', 'does', 'did', 'is', 'are', 'can', 'could', 'should')
    is_question = ('?' in ui) or ui.split()[0] in interrogatives if ui else False
    
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', rt) if s.strip()]
    if not sentences:
        return "I understand. How else can I help?"
        
    # For questions, prioritize answer-like sentences
    if is_question:
        # Look for sentences that seem like answers
        for s in sentences:
            # Prefer sentences with relevant content
            if any(w in s.lower() for w in ui.split()):
                return s if s[-1] in '.!?' else s + '.'
        # Fallback to first sentence
        return sentences[0] if sentences[0][-1] in '.!?' else sentences[0] + '.'

    # For statements, keep response focused and concise
    first = sentences[0]
    words = first.split()
    if len(words) > 30:
        first = ' '.join(words[:30]).rstrip(' ,;:') + '...'
    if first[-1] not in '.!?':
        first += '.'
    return first

if __name__ == '__main__':
    print("🧠 Покров Диалоговая Система")
    print_model_info()
    responder = build_chat_responder(max_examples=500)
    
    context = []  # Store conversation context
    last_resp = None  # Track last response for anti-repeat
    
    while True:
        user_input = input("🗨 Ты: ")
        if user_input.lower() in ["exit", "quit", "выход"]:
            break

        # Encode user input with context
        if tokenizer is None:
            from phase_tokenizer import PhaseTokenizer
            tokenizer = PhaseTokenizer(dim=128, h=0.05, i=1.0)
            
        # Include recent context in encoding
        context_str = " ".join(context[-3:] + [user_input])  # Last 3 turns + current
        phase_input = tokenizer.encode_text(context_str)
        
        # Get phase embedding
        dummy_label = torch.tensor([0])
        if a_temp is None:
            a_temp = torch.full((1, 1), 0.5, dtype=torch.float32)
        with torch.no_grad():
            _, input_phase = model(phase_input, a_temp, dummy_label, dummy_label, epoch=999)

        # Get candidate responses
        cand_phases = torch.stack([e['phase'] for e in responder.entries], dim=0)
        cand_texts = [_stringify_response(e.get('response')) for e in responder.entries]
        
        # Enhanced response selection considering only user input
        selection = select_response(input_phase.squeeze(0), cand_phases, cand_texts, user_input)
        
        if selection['is_direct'] and selection['sim'] > 0.85:
            # Use direct response for high confidence matches
            resp_text = cand_texts[selection['index']]
        else:
            # Otherwise use phase composition with only user input
            reply = compose_reply_from_responder(
                input_phase.squeeze(0),
                responder,
                k=5,
                temperature=0.6,
                out_len=100,
                user_text=user_input  # Use only current user input
            )
            resp_text = reply['composed_text']

        # Post-process for relevance and conciseness
        resp_text = make_direct_reply(user_input, resp_text)
        
        # Anti-repeat check - try higher temperature/diversity if same as last response
        if resp_text == last_resp:
            print("[ANTI_REPEAT] detected same reply as previous — trying higher temperature fallback")
            fallback = compose_reply_from_responder(
                input_phase.squeeze(0),
                responder,
                k=10,  # More candidates
                temperature=1.0,  # Higher temperature
                out_len=150,  # Allow longer response
                user_text=user_input
            )
            resp_text = make_direct_reply(user_input, fallback.get('composed_text', '') or resp_text)
        
        # Update tracking
        last_resp = resp_text
        
        # Update context
        context.append(user_input)
        context.append(resp_text)
        if len(context) > 6:  # Keep last 3 turns (6 messages)
            context = context[-6:]
            
        print(f"Архитектор: {resp_text}")
