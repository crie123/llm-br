import sys
sys.path.append("project_libs")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from plotting import plot_all

# Define the domain
x1 = np.linspace(-1.5, 1.5, 200)
x2 = np.linspace(-1, 1.5, 200)
x3 = np.linspace(-1, 1, 200)

# Function 1: Exponentially decreases then increases (symmetric tanh)
def f1(x):
    return -np.tanh(2 * x)

# Function 2: Exponential rise and fall, then linear
def f2(x):
    return np.piecewise(x, [x <= 1, x > 1], [lambda x: np.tanh(2 * x), lambda x: 0.07 * x + 0.894])

# Function 3: Linearly decreases and increases, smoothed with tanh near joins
def f3(x):
    return np.piecewise(x, [x <= 0, x > 0], [lambda x: -0.5 * x - 0.5 * np.tanh(3 * (x + 0.5)), lambda x: 0.5 * x - 0.5 * np.tanh(3 * (x - 0.5))])
# FlexiblePReLU class
class FlexiblePReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(0.50))  # learnable scalar

    def forward(self, x, a):
        lower = a - 0.5
        upper = a + 0.5
        in_region = (a >= -1) & (a <= 1) & (x > lower) & (x < upper)

        out = torch.where(in_region, x * self.r, x)
        return out

# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dtype=torch.float32):
        super(NeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Custom functions
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.flexible_prelu1 = FlexiblePReLU()
        self.flexible_prelu2 = FlexiblePReLU()
        self.flexible_prelu3 = FlexiblePReLU()
        self.fc = nn.Linear(hidden_size, output_size)  # Final linear layer
    
    def forward(self, X, a1, a2, a3):
        X = self.embedding(X)
        X = X.mean(dim=1)  # Average pooling over the token dimension
        X = self.fc(X)  # Pass through the final linear layer
        z1 = torch.tensor(self.f1(X.detach().numpy()), dtype=torch.float32)
        a1 = self.flexible_prelu1(z1, a1)
        z2 = torch.tensor(self.f2(a1.detach().numpy()), dtype=torch.float32)
        a2 = self.flexible_prelu2(z2, a2)
        z3 = torch.tensor(self.f3(a2.detach().numpy()), dtype=torch.float32)
        a3 = self.flexible_prelu3(z3, a3)
        return a3  # Return logits for CrossEntropyLoss

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
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(nn.parameters(), lr=0.001)

# Individual weights for each layer
a1 = torch.tensor(np.full((X.size(0), 1), 0.5), dtype=torch.float32)  # Example weight for fc1
a2 = torch.tensor(np.full((X.size(0), 1), 0.5), dtype=torch.float32)  # Example weight for fc2
a3 = torch.tensor(np.full((X.size(0), 1), 0.5), dtype=torch.float32)  # Example weight for fc3

# Store metrics for plotting
W1_history = []
loss_history = []
accuracy_history = []

for epoch in range(10):
    optimizer.zero_grad()
    output = nn(X, a1, a2, a3)
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

print(f"x1 shape: {x1.shape}")
print(f"x2 shape: {x2.shape}")
print(f"x3 shape: {x3.shape}")
print(f"f1 shape: {f1(x1).shape}")
print(f"f2 shape: {f2(x2).shape}")
print(f"f3 shape: {f3(x3).shape}")
print(f"Number of unique emotions: {len(ds['train'].unique('emotion'))}")
print(f"Output shape: {output.shape}")
print(f"Predicted classes shape: {predicted_classes.shape}")
print(f"y shape: {y.shape}")
plot_all(W1_history, x1, x2, x3, f1, f2, f3, X.numpy(), y.numpy(), output.detach().numpy().squeeze(), loss_history, accuracy_history)