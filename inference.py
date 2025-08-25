import torch
import numpy as np
from neural_network import NeuralNetwork, EmpathicDatasetResponder
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

# LOAD MODEL 
checkpoint = torch.load("pokrov_model.pt")
model = NeuralNetwork(
    input_size=128,      
    hidden_size=32,
    output_size=32,      
    num_layers=20        
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# LOAD DATASET 
ds = load_dataset("Estwld/empathetic_dialogues_llm")
from phase_tokenizer import PhaseTokenizer
tokenizer = PhaseTokenizer(dim=128, h=0.05, i=1.0)
bert_model = None

def embed_conversation(examples):
    texts = []
    for conv in examples["conversations"]:
        # conv may be a list of strings or a list of dicts (dataset-dependent)
        if isinstance(conv, list):
            parts = []
            for turn in conv:
                if isinstance(turn, str):
                    parts.append(turn)
                elif isinstance(turn, dict):
                    # try common keys, otherwise join all values
                    for k in ("text", "utterance", "message", "content", "sentence"):
                        if k in turn:
                            parts.append(str(turn[k]))
                            break
                    else:
                        parts.append(" ".join(str(v) for v in turn.values()))
                else:
                    parts.append(str(turn))
            t = " ".join(parts)
        else:
            t = str(conv)
        texts.append(t)
    embeds = [tokenizer.encode_text(t).squeeze(0).numpy() for t in texts]
    return {"bert_embed": np.stack(embeds).astype(np.float32)}

ds = ds.map(embed_conversation, batched=True, batch_size=8)

# DUMMY EMBEDDINGS (if already preprocessed) 
X = torch.tensor(ds["train"]["bert_embed"], dtype=torch.float32)
all_emotions = ds["train"]["emotion"]
label_encoder = LabelEncoder()
label_encoder.fit(all_emotions)

a_temp = torch.full((1, 1), 0.5, dtype=torch.float32)

# BUILD PHASE-BASED RESPONDER 
responder = EmpathicDatasetResponder(dataset=ds["train"],label_encoder=label_encoder,bert_model=bert_model,tokenizer=tokenizer,model=model,a_temp=a_temp)

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

# CHAT LOOP 
print("🧠 Покров Диалоговая Система")
while True:
    user_input = input("🗨 Ты: ")
    if user_input.lower() in ["exit", "quit", "выход"]:
        break

    # Temporary embedding for user input
    phase_input = tokenizer.encode_text(user_input)
    dummy_embed = phase_input
    dummy_label = torch.tensor([0])
    with torch.no_grad():
        _, input_phase = model(dummy_embed, a_temp, dummy_label, dummy_label, epoch=999)
    print(f"Loaded {len(responder.entries)} entries.")
    reply = responder.reply_from_data(input_phase.squeeze(0))

    # Normalize reply into a readable string
    resp_text = ""
    tag_text = ""
    if isinstance(reply, tuple) and len(reply) >= 1:
        resp, *rest = reply
        tag_text = rest[0] if rest else ""
    else:
        resp = reply

    if isinstance(resp, list):
        parts = []
        for turn in resp:
            if isinstance(turn, dict):
                # prefer assistant content, then any content/text
                if turn.get('role') == 'assistant' and 'content' in turn:
                    parts.append(str(turn['content']))
                elif 'content' in turn:
                    parts.append(str(turn['content']))
                elif 'text' in turn:
                    parts.append(str(turn['text']))
                else:
                    parts.append(" ".join(str(v) for v in turn.values()))
            else:
                parts.append(str(turn))
        resp_text = " ".join(p for p in parts if p)
    elif isinstance(resp, dict):
        resp_text = str(resp.get('content') or resp.get('text') or resp)
    else:
        resp_text = str(resp)

    out = resp_text
    if tag_text:
        out = f"{out} {tag_text}"

    print(f"🤖 Архитектор: {out}")
