import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
import numpy as np

def plot_all(W1_history, x1, x2, x3, f1, f2, f3, X, y, output, loss_history, accuracy_history):
    fig = plt.figure(figsize=(24, 12))
    
    # Function 1
    plt.subplot(2, 3, 1)
    plt.plot(x1, f1(x1), label="f1 (exp. decrease increase)", color='blue')
    plt.title("Function 1")
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Function 2
    plt.subplot(2, 3, 2)
    plt.plot(x2, f2(x2), label="f2 (exp. + linear)", color='green')
    plt.title("Function 2")
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Function 3
    plt.subplot(2, 3, 3)
    plt.plot(x3, f3(x3), label="f3 (smooth linear)", color='red')
    plt.title("Function 3")
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Confusion Matrix
    plt.subplot(2, 3, 4)
    cm = confusion_matrix(y, np.argmax(output, axis=1))
    classes = unique_labels(y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=plt.gca(), cmap='viridis', colorbar=False)
    plt.title('Confusion Matrix')
    
    # Training Loss
    plt.subplot(2, 3, 5)
    plt.plot(loss_history, label="Training Loss", color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch (x1000)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Training Accuracy
    plt.subplot(2, 3, 6)
    plt.plot(accuracy_history, label="Training Accuracy", color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch (x1000)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()