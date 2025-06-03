import sys
sys.path.append("project_libs")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from plotting import plot_all
from functions import f1, f2, f3

# Emotions and their corresponding character sectors
emotion_to_sector = {
    "afraid": "melancholic", "angry": "choleric", "annoyed": "choleric",
    "anticipating": "sanguine", "anxious": "melancholic", "apprehensive": "melancholic",
    "confident": "sanguine", "content": "phlegmatic", "devastated": "melancholic",
    "disappointed": "melancholic", "embarrassed": "melancholic", "excited": "sanguine",
    "faithful": "phlegmatic", "grateful": "phlegmatic", "guilty": "melancholic",
    "hopeful": "sanguine", "impressed": "sanguine", "jealous": "choleric",
    "joyful": "sanguine", "lonely": "melancholic", "nostalgic": "phlegmatic",
    "proud": "sanguine", "sad": "melancholic", "terrified": "melancholic",
    "trusting": "phlegmatic"
}

sector_to_weight = {
    "melancholic": 0.92,
    "choleric": 1.08,
    "phlegmatic": 0.98,
    "sanguine": 1.05,
}

# FlexiblePReLU class
class FlexiblePReLU(nn.Module):
    def __init__(self, sector_weights):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(0.09, dtype=torch.float32))
        self.sector_weights = sector_weights

    def forward(self, x, a, emotion_labels):
        lower = a - 0.5
        upper = a + 0.5
        in_region = (a >= -1) & (a <= 1) & (x > lower) & (x < upper)

        emotion_multipliers = torch.tensor(
            [self.sector_weights.get(emotion_to_sector.get(em, ''), 1.0) for em in emotion_labels],
            dtype=torch.float32,
            device=x.device
        ).unsqueeze(1)

        out = torch.where(in_region, x * self.r * emotion_multipliers, x)
        return out

# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=20, dtype=torch.float32):
        super(NeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.flexible_prelu_layers = nn.ModuleList([
            FlexiblePReLU(sector_weights=sector_to_weight) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.functions = [self.f1, self.f2, self.f3]
        self.num_layers = num_layers

        # Emotion-based parameter generator
        self.a_table = nn.Embedding(output_size, num_layers)

    def forward(self, X, emotion_ids, emotion_labels):
        X = self.embedding(X)
        X = X.mean(dim=1)
        emotion_ids = emotion_ids.long()
        a = self.a_table(emotion_ids)  # shape: [batch_size, num_layers]
        
        for i, layer in enumerate(self.hidden_layers):
            X = layer(X)
            func = self.functions[i % 3]
            z = torch.tensor(func(X.detach().cpu().numpy()), dtype=torch.float32, device=X.device)
            
            # Ensure `a` matches the batch size of `X`
            a_layer = a[:, i].unsqueeze(1)  # Select the i-th layer weights for the batch
            X = self.flexible_prelu_layers[i](z, a_layer, emotion_labels)
        
        return self.fc(X)

# Load the dataset
from datasets import load_dataset, Dataset
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

ds = load_dataset("Estwld/empathetic_dialogues_llm")

# Tokenize the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    conversations = examples['conversations']
    conversation_texts = [' '.join([conv['content'] for conv in conversation]) for conversation in conversations]
    return tokenizer(conversation_texts, padding='max_length', truncation=True)

ds = ds.map(tokenize_function, batched=True)

# Convert to PyTorch tensors
X = torch.tensor(ds['train']['input_ids'], dtype=torch.long)
# Encode emotion labels
label_encoder = LabelEncoder()
ds = ds.map(lambda x: {'emotion': label_encoder.fit_transform(x['emotion'])}, batched=True)

y = torch.tensor(ds['train']['emotion'], dtype=torch.long)  # Use 'emotion' as the label

# Create and train the neural network
input_size = tokenizer.vocab_size
hidden_size = 128
output_size = len(ds['train'].unique('emotion'))  # Multi-class classification
nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=20, dtype=torch.float32)
# Calculate class weights
class_counts = torch.bincount(y)
class_weights = 0.5 / class_counts.float()
class_weights = class_weights / class_weights.sum()

# Initialize CrossEntropyLoss with class weights
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(nn.parameters(), lr=0.001)

# Individual weights for each layer
batch_size = X.size(0)  # Recieve batch size from input data
num_layers = 20  # Number of layers in the neural network
a = torch.full((batch_size, num_layers), 0.5, dtype=torch.float32)  # Creating a tensor of shape (batch_size, num_layers) filled with 0.5

# Store metrics for plotting
W1_history = []
loss_history_f1 = []
accuracy_history_f1 = []
loss_history_f2 = []
accuracy_history_f2 = []
loss_history_f3 = []
accuracy_history_f3 = []

for epoch in range(1000):
    optimizer.zero_grad()
    output = nn(X, y, y)  # Passing y as emotion_ids and labels
    loss = criterion(output, y)  # Correct shape for CrossEntropyLoss
    predicted_classes = torch.argmax(output, dim=1)  # Convert logits to class predictions
    correct = (predicted_classes == y).sum().item()
    accuracy = correct / y.size(0)
    accuracy_history_f1.append(accuracy)
    accuracy_history_f2.append(accuracy)
    accuracy_history_f3.append(accuracy)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    loss_history_f1.append(loss.item())
    loss_history_f2.append(loss.item())
    loss_history_f3.append(loss.item())

# Create and use the Plotter class
from plotting import plot_all

print(f"Number of unique emotions: {len(ds['train'].unique('emotion'))}")
print(f"Output shape: {output.shape}")
print(f"Predicted classes shape: {predicted_classes.shape}")
print(f"y shape: {y.shape}")
plot_all(
    y.numpy(),
    output.detach().numpy().squeeze(),
    loss_history_f1,
    accuracy_history_f1,
    loss_history_f2,
    accuracy_history_f2,
    loss_history_f3,
    accuracy_history_f3
)