import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
import numpy as np

def plot_all(y, output, loss_history, accuracy_history):
    fig = plt.figure(figsize=(24, 12))
    
    
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