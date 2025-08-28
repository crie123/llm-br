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
    return idx, vals

class PhaseComposer:
    def __init__(self, dim: int = 128):
        self.dim = dim

    @staticmethod
    def softmax(x: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        x = x / max(t, 1e-6)
        x = x - x.max()
        return torch.exp(x) / (torch.exp(x).sum() + 1e-8)

    def compose(self, phases: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        phi = (phases.clamp(-1, 1) * _math.pi)
        z = torch.exp(1j * phi)
        w = weights.view(-1, 1)
        z_mix = (w * z).sum(dim=0)
        comp_phi = torch.angle(z_mix)
        comp_norm = torch.tensor(np.interp(comp_phi.cpu().numpy(), (-_math.pi, _math.pi), (-1, 1)), dtype=torch.float32, device=phases.device)
        return comp_norm

    def decode_to_text(self, phase: torch.Tensor, out_len: int = 64, guide_texts: List[str] = None) -> str:
        """Decode phase into a short readable ASCII string.
        We generate a deterministic sequence by iFFT, normalize to [0,1], then map to a small printable charset.
        If guide_texts is provided we still prefer a reasonably long real candidate; otherwise return the decoded sketch.
        """
        try:
            phi = (phase.clamp(-1, 1) * math.pi).cpu().numpy()
        except Exception:
            # handle plain numpy arrays or lists
            phi = (np.clip(np.array(phase), -1, 1) * math.pi)
        
        spec = np.exp(1j * phi)
        seq = np.fft.ifft(spec).real
        # normalize to [0,1]
        rng = seq.max() - seq.min()
        if rng <= 1e-6:
            norm = np.zeros_like(seq)
        else:
            norm = (seq - seq.min()) / rng

        # choose a small readable charset
        charset = list("abcdefghijklmnopqrstuvwxyz0123456789 ,.!?')(")
        L = len(charset)
        indices = (norm * (L - 1)).astype(int)[:out_len]
        chars = [charset[i] for i in indices]
        s = ''.join(chars)
        # collapse repeated characters and trim whitespace
        s = re.sub(r"(\s){2,}", " ", s)
        s = re.sub(r"(.)\1{2,}", r"\1\1", s)
        s = s.strip(" '\"")
        # make sure it ends with punctuation; if not, add a period
        if s and s[-1] not in '.!?':
            s = s + '.'
        decoded = ' '.join(s.split())

        # If guidance texts available, prefer them when they look informative
        if guide_texts:
            candidates = [t for t in guide_texts if t and isinstance(t, str)]
            if candidates:
                best = max(candidates, key=lambda x: len(x.split()))
                if len(best) > max(20, out_len // 2):
                    return best
                # otherwise combine short guide and decoded sketch for readability
                merged = (best + ' — ' + decoded).strip()
                return ' '.join(merged.split())

        return decoded

    def compose_text(self,
                     query_phase: torch.Tensor,
                     cand_phases: torch.Tensor,
                     cand_texts: List[str],
                     temperature: float = 0.5,
                     k: int = 5,
                     out_len: int = 160) -> Dict[str, Any]:
        idx, sims = topk_by_phase(query_phase, cand_phases, k=k)
        w = self.softmax(sims, t=temperature)
        mixed = self.compose(cand_phases[idx], w)
        # pass picked texts as guidance to the decoder for more coherent speech
        picked = [cand_texts[i] for i in idx.tolist()] if cand_texts else []
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

def compose_reply_from_responder(query_phase: torch.Tensor, responder_obj, k: int = 5, temperature: float = 0.6, out_len: int = 100) -> Dict[str, Any]:
    if not getattr(responder_obj, 'entries', None):
        return {'composed_text': '', 'picked_texts': [], 'indices': [], 'sims': [], 'weights': []}

    cand_phases = torch.stack([e['phase'] for e in responder_obj.entries], dim=0)
    cand_texts = [_stringify_response(e.get('response')) for e in responder_obj.entries]

    # prefer a single direct candidate reply instead of merging many dataset turns
    qp = query_phase.squeeze(0) if query_phase.dim() == 2 else query_phase
    # compute cosine similarities and pick best
    sims = torch.nn.functional.cosine_similarity(qp.unsqueeze(0), cand_phases, dim=1)
    best_idx = int(torch.argmax(sims).item())
    best_sim = float(sims[best_idx].item())

    # if top candidate is reasonably similar, return it trimmed to short, relevant snippet
    SIM_THRESHOLD = 0.15
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
        if len(words) > max_words:
            out = ' '.join(words[:max_words]).rstrip(' ,;:') + '...'
        return out

    if best_sim >= SIM_THRESHOLD:
        # Return the dataset candidate directly (trimmed). This is more reliable than
        # decoding its phase which may yield token-like gibberish.
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

    # otherwise fallback to composing a fresh phase.
    # Provide the top dataset candidate as lightweight guidance (trimmed) so the decoder
    # produces readable, context-aligned text instead of token-like gibberish.
    top_candidate = cand_texts[best_idx]
    top_trim = trim_reply(top_candidate, max_sentences=3, max_words=60)
    guide = [top_trim] if top_trim else []
    comp = composer.compose_text(qp, cand_phases, guide, temperature=temperature, k=min(k, cand_phases.size(0)), out_len=out_len)
    # ensure composed text is concise
    comp_text = trim_reply(comp.get('composed_text', ''), max_sentences=3)
    comp['composed_text'] = comp_text
    return comp

def build_chat_responder(chat_dataset_name: str = "daily_dialog", max_examples: int = 1000):
    from neural_network import EmpathicDatasetResponder
    # create tokenizer and a_temp lazily if not already available
    global tokenizer, a_temp, _label_encoder
    if tokenizer is None:
        from phase_tokenizer import PhaseTokenizer
        tokenizer = PhaseTokenizer(dim=128, h=0.05, i=1.0)
    if a_temp is None:
        a_temp = torch.full((1, 1), 0.5, dtype=torch.float32)

    ds_chat = None
    try:
        ds_chat = load_dataset(chat_dataset_name, split='train')
    except ValueError as e:
        msg = str(e)
        if 'trust_remote_code' in msg:
            # retry allowing local/custom dataset code to run (safe for local workspace)
            try:
                ds_chat = load_dataset(chat_dataset_name, split='train', trust_remote_code=True)
            except Exception:
                ds_chat = None
        else:
            ds_chat = None
    except Exception:
        # fallback: try loading without split
        try:
            ds_chat = load_dataset(chat_dataset_name)
        except Exception:
            ds_chat = None

    if ds_chat is None:
        # dataset couldn't be loaded; proceed with empty dataset (no local placeholder injections)
        print(f"[ChatResponder] couldn't load '{chat_dataset_name}' — proceeding with empty dataset (no placeholders).")
        ds_small = []
    else:
        ds_small = ds_chat.select(list(range(min(len(ds_chat), max_examples))))

    # build a simple label encoder if not present
    if _label_encoder is None:
        _label_encoder = LabelEncoder()
        _label_encoder.fit(["neutral"])
    responder_chat = EmpathicDatasetResponder(dataset=ds_small, label_encoder=_label_encoder, bert_model=None, tokenizer=tokenizer, model=model, a_temp=a_temp)
    # If responder couldn't extract any entries, warn and return the responder as-is (no placeholder injection).
    if not getattr(responder_chat, 'entries', None) or len(responder_chat.entries) == 0:
        print("[ChatResponder] warning: constructed responder has 0 entries — continuing without placeholders.")
    return responder_chat

def evaluate_bitgrid_recognition(responder_obj, num_images: int = 200, grid: int = 16, subset: str = "train[:%d]" % 200):
    """Load CIFAR10 subset, for each image compute ground-truth bitgrid and query composer with image phase.
    Measure IoU and bit accuracy between composed phase->bitgrid and true image bitgrid. Plot distributions.
    """
    try:
        # BitGridSensor imported from bitgrid above
        sensor = BitGridSensor(dim=128, grid=grid, threshold=0.5)
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
    - If user asked a question (contains '?', or starts with interrogative words), return first sentence of reply_text.
    - If user greeted, return a short greeting.
    - Otherwise, return up to 25 words from the first sentence of reply_text.
    """
    import re
    if not reply_text or not isinstance(reply_text, str):
        return "Sorry, I don't have an answer right now."

    ui = (user_input or "").strip().lower()
    # greetings
    if re.match(r'^(hi|hello|hey|привет|здравствуй|здравствуйте)\b', ui):
        # prefer a short greeting from reply_text if present
        first = re.split(r'(?<=[.!?])\s+', reply_text.strip())[0]
        if len(first.split()) <= 6:
            return first
        return "Hi — how can I help?"

    # question detection
    interrogatives = ('who', 'what', 'when', 'where', 'why', 'how', 'do', 'does', 'did', 'is', 'are', 'can', 'could', 'should')
    is_question = ('?' in ui) or ui.split()[0] in interrogatives if ui else False

    # take first sentence
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', reply_text.strip()) if s.strip()]
    first = sentences[0] if sentences else reply_text.strip()

    # If it's a question, be concise
    if is_question:
        # keep only first sentence and cap words
        words = first.split()
        if len(words) > 40:
            first = ' '.join(words[:40]).rstrip(' ,;:') + '...'
        # ensure it ends with punctuation
        if first[-1] not in '.!?':
            first = first + '.'
        return first

    # otherwise, return a short fragment (up to 25 words)
    words = first.split()
    if len(words) > 40:
        first = ' '.join(words[:40]).rstrip(' ,;:') + '...'
    return first

if __name__ == '__main__':
    # CHAT LOOP (moved under main guard). Build responder lazily to avoid heavy work during import.
    print("🧠 Покров Диалоговая Система")
    print_model_info()
    responder = build_chat_responder(max_examples=500)
    while True:
        user_input = input("🗨 Ты: ")
        if user_input.lower() in ["exit", "quit", "выход"]:
            break

        # Temporary embedding for user input
        if tokenizer is None:
            from phase_tokenizer import PhaseTokenizer
            tokenizer = PhaseTokenizer(dim=128, h=0.05, i=1.0)
        phase_input = tokenizer.encode_text(user_input)
        dummy_embed = phase_input
        dummy_label = torch.tensor([0])
        if a_temp is None:
            a_temp = torch.full((1, 1), 0.5, dtype=torch.float32)
        with torch.no_grad():
            _, input_phase = model(dummy_embed, a_temp, dummy_label, dummy_label, epoch=999)
        print(f"Loaded {len(responder.entries)} entries.")
        reply = compose_reply_from_responder(input_phase.squeeze(0), responder, k=5, temperature=0.6, out_len=100)

        # Normalize reply into a readable string
        resp_text = reply['composed_text']
        # Post-process to make response directly relevant and concise
        resp_text = make_direct_reply(user_input, resp_text)
        tag_text = ""

        out = resp_text
        if tag_text:
            out = f"{out} {tag_text}"

        print(f"Архитектор: {out}")
