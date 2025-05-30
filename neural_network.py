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

# FlexiblePReLU class
class FlexiblePReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(0.20))  # learnable scalar

    def forward(self, x, a):
        lower = a - 0.5
        upper = a + 0.5
        in_region = (a >= -1) & (a <= 1) & (x > lower) & (x < upper)

        out = torch.where(in_region, x * self.r, x)
        return out

# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=10, dtype=torch.float32):
        super(NeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.flexible_prelu_layers = nn.ModuleList([FlexiblePReLU() for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)  # Final linear layer
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

    def forward(self, X, a):
        X = self.embedding(X)
        X = X.mean(dim=1)  # Average pooling over the token dimension
        for i, layer in enumerate(self.hidden_layers):
            X = layer(X)
            if i == 0:
                z = torch.tensor(self.f1(X.detach().numpy()), dtype=torch.float32)
                X = self.flexible_prelu_layers[i](z, a[:, i].unsqueeze(1))
            elif i == 1:
                z = torch.tensor(self.f2(X.detach().numpy()), dtype=torch.float32)
                X = self.flexible_prelu_layers[i](z, a[:, i].unsqueeze(1))
            elif i == 2:
                z = torch.tensor(self.f3(X.detach().numpy()), dtype=torch.float32)
                X = self.flexible_prelu_layers[i](z, a[:, i].unsqueeze(1))
            else:
                X = self.flexible_prelu_layers[i](X, a[:, i].unsqueeze(1))
        X = self.fc(X)  # Pass through the final linear layer
        return X  # Return the output of the final linear layer for loss calculation

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
nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dtype=torch.float32)
# Calculate class weights
class_counts = torch.bincount(y)
class_weights = 0.5 / class_counts.float()
class_weights = class_weights / class_weights.sum()

# Initialize CrossEntropyLoss with class weights
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(nn.parameters(), lr=0.001)

# Individual weights for each layer
a = torch.tensor(np.full((X.size(0), 10), 0.1), dtype=torch.float32)  # Example weight for each layer

# Store metrics for plotting
W1_history = []
loss_history = []
accuracy_history = []

for epoch in range(10):
    optimizer.zero_grad()
    output = nn(X, a)
    loss = criterion(output, y)  # Correct shape for CrossEntropyLoss
    predicted_classes = torch.argmax(output, dim=1)  # Convert logits to class predictions
    correct = (predicted_classes == y).sum().item()
    accuracy = correct / y.size(0)
    accuracy_history.append(accuracy)
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    loss_history.append(loss.item())

# Create and use the Plotter class
from plotting import plot_all

print(f"Number of unique emotions: {len(ds['train'].unique('emotion'))}")
print(f"Output shape: {output.shape}")
print(f"Predicted classes shape: {predicted_classes.shape}")
print(f"y shape: {y.shape}")
plot_all(y.numpy(), output.detach().numpy().squeeze(), loss_history, accuracy_history)