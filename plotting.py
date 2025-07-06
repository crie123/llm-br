import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay

def plot_phase_map(x_tensor, y_tensor, i, h, title="Phase Error Map"):
    
    x = x_tensor.detach().cpu().numpy().flatten()
    y = y_tensor.detach().cpu().numpy().flatten()

    size = min(len(x), 1000)
    x = x[:size]
    y = y[:size]

    X, Y = np.meshgrid(x, y)
    Z = i * np.sin(np.pi * X * Y * h)

    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel("Input X")
    plt.ylabel("Target Y")
    plt.show()

def plot_all(y, output, loss_history_f1, accuracy_history_f1, loss_history_f2, accuracy_history_f2, loss_history_f3, accuracy_history_f3):
    # Print Training Loss History
    print("Training Loss History:")
    for epoch, loss in enumerate(loss_history_f1):
        print(f"Epoch {epoch + 1}: Loss = {loss}")
    
    # Print Training Accuracy History
    print("Training Accuracy History:")
    for epoch, accuracy in enumerate(accuracy_history_f1):
        print(f"Epoch {epoch + 10}: Accuracy = {accuracy}")
    
    # Plot Loss History for f1
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.plot(loss_history_f1, label='f1')
    plt.title('Loss History (f1)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Loss History for f2
    plt.subplot(2, 3, 2)
    plt.plot(loss_history_f2, label='f2')
    plt.title('Loss History (f2)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Loss History for f3
    plt.subplot(2, 3, 3)
    plt.plot(loss_history_f3, label='f3')
    plt.title('Loss History (f3)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy History for f1
    plt.subplot(2, 3, 4)
    plt.plot(accuracy_history_f1, label='f1')
    plt.title('Accuracy History (f1)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Accuracy History for f2
    plt.subplot(2, 3, 5)
    plt.plot(accuracy_history_f2, label='f2')
    plt.title('Accuracy History (f2)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Accuracy History for f3
    plt.subplot(2, 3, 6)
    plt.plot(accuracy_history_f3, label='f3')
    plt.title('Accuracy History (f3)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # F1-score
    predicted_labels = np.argmax(output, axis=1)
    f1 = f1_score(y, predicted_labels, average='weighted')
    print(f"F1 Score: {f1}")

    # MSE
    mse = mean_squared_error(y, predicted_labels)
    print(f"Mean Squared Error: {mse}")

    # Confusion matrix
    cm = confusion_matrix(y, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis')
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.show()

    # Check if training stopped after 100 epochs
    if len(loss_history_f1) > 100:
        print("Training stopped after 100 epochs.")
    else:
        print("Training did not stop after 100 epochs.")
    
    # Additional analysis to determine why training might have stopped
    if len(loss_history_f1) > 100:
        last_100_losses = loss_history_f1[-100:]
        if np.allclose(last_100_losses, last_100_losses[0], atol=1e-3):
            print("Training stopped because the loss did not change significantly over the last 100 epochs.")
        else:
            print("Training stopped for an unknown reason.")
    else:
        print("No additional analysis available as training did not reach 100 epochs.")