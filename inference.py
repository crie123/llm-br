import torch
from transformers import BertTokenizer, BertModel
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
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

def embed_conversation(examples):
    batch_size = len(examples["conversations"])
    dummy_embed = np.random.normal(0, 1, size=(batch_size, 128))
    return {"bert_embed": dummy_embed.astype(np.float32)}

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
    dummy_embed = torch.randn(1, checkpoint["input_size"])
    dummy_label = torch.tensor([0])
    with torch.no_grad():
        _, input_phase = model(dummy_embed, a_temp, dummy_label, dummy_label, epoch=999)
    print(f"Loaded {len(responder.entries)} entries.")
    reply = responder.reply_from_data(input_phase.squeeze(0))
    print(f"🤖 Покров: {reply}")
