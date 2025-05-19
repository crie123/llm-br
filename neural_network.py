import sys
sys.path.append("project_libs")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from plotting import plot_functions, plot_confusion_matrix, plot_training

# Define the domain
x1 = np.linspace(-1.5, 1.5, 400)
x2 = np.linspace(-1, 1.5, 400)
x3 = np.linspace(-1, 1, 400)

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
        self.r = nn.Parameter(torch.tensor(0.25) + torch.tensor(np.full((400, 1), 0.5), dtype=torch.float64))  # learnable scalar

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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = nn.Parameter(torch.randn(self.input_size, self.hidden_size, dtype=dtype))
        self.b1 = nn.Parameter(torch.zeros(1, self.hidden_size, dtype=dtype))
        self.W2 = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, dtype=dtype))
        self.b2 = nn.Parameter(torch.zeros(1, self.hidden_size, dtype=dtype))
        self.W3 = nn.Parameter(torch.randn(self.hidden_size, self.output_size, dtype=dtype))
        self.b3 = nn.Parameter(torch.zeros(1, self.output_size, dtype=dtype))
        
        # FlexiblePReLU layers
        self.flexible_prelu1 = FlexiblePReLU()
        self.flexible_prelu2 = FlexiblePReLU()
        self.flexible_prelu3 = FlexiblePReLU()
    
    def forward(self, X, a1, a2, a3):
        z1 = torch.matmul(X, self.W1) + self.b1
        a1 = self.flexible_prelu1(z1, a1)
        
        z2 = torch.matmul(a1, self.W2) + self.b2
        a2 = self.flexible_prelu2(z2, a2)
        
        z3 = torch.matmul(a2, self.W3) + self.b3
        a3 = self.flexible_prelu3(z3, a3)
        
        return a3  # Return logits for CrossEntropyLoss

# Generate sample data
X = torch.tensor(np.array([x1, x2, x3]).T, dtype=torch.float64)
y_values = np.array([f1(x1), f2(x2), f3(x3)]).T
y_classes = np.argmax(y_values, axis=1)  # Convert to class indices
y = torch.tensor(y_classes, dtype=torch.long)

# Create and train the neural network
nn = NeuralNetwork(input_size=3, hidden_size=14, output_size=3, dtype=torch.float64)  # Increased hidden size to 14
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(nn.parameters(), lr=0.1)

# Individual weights for each function
a1 = torch.tensor(np.full((400, 1), 0.5), dtype=torch.float64)  # Example weight for f1
a2 = torch.tensor(np.full((400, 1), 0.5), dtype=torch.float64)  # Example weight for f2
a3 = torch.tensor(np.full((400, 1), 0.5), dtype=torch.float64)  # Example weight for f3

# Store metrics for plotting
W1_history = []
loss_history = []
accuracy_history = []

for epoch in range(10000):
    optimizer.zero_grad()
    output = nn(X, a1, a2, a3)
    loss = criterion(output, y)
    predicted_classes = output.argmax(dim=1)
    correct = (predicted_classes == y).sum().item()
    accuracy = correct / y.size(0)
    accuracy_history.append(accuracy)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    W1_history.append(nn.W1.detach().numpy().copy())
    loss_history.append(loss.item())

# Create and use the Plotter class
plotter = Plotter(x1, x2, x3, f1, f2, f3, X, y, output, W1_history)
plotter.plot_confusion_matrix(y, output.argmax(dim=1))
plotter.plot_training(loss_history, accuracy_history)