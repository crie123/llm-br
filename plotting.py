import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_functions(W1_history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    current_plot = 0

    ax_button = fig.add_axes([0.8, 0.05, 0.1, 0.075])
    button = plt.Button(ax_button, 'Next')
    button.on_clicked(next_plot)
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

def plot_training(loss_history, accuracy_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch (x1000)')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch (x1000)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()