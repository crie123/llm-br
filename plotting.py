import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_phase_map(x_tensor, y_tensor, i, h, title="Phase Error Map"):
    x = x_tensor.detach().cpu().numpy().flatten()
    y = y_tensor.detach().cpu().numpy().flatten()

    size = min(len(x), 128)  # limit size to 128
    x = x[:size]
    y = y[:size]

    X, Y = np.meshgrid(x, y)
    Z = i * np.sin(np.pi * X * Y * h)

    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')  # levels limited to 50 for better visibility
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel("Input X")
    plt.ylabel("Target Y")
    plt.show()

def plot_consciousness_metrics(archsim, rcl, drift, score):
    plt.figure(figsize=(10, 4))
    plt.plot(archsim, label="ArchSim")
    plt.title("Phase ArchSim Over Time")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(rcl, label="Clarity")
    plt.plot(drift, label="Drift")
    plt.title("Drift vs RCL")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(score, label="PhaseConsciousnessScore")
    plt.title("Pokrov Consciousness Score Over Time")
    plt.legend()
    plt.show()

def plot_resonant_clusters_matrix(y_true, cluster_labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, cluster_labels)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Resonant Clusters Matrix")
    plt.xlabel("Cluster ID")
    plt.ylabel("True Emotion Label")
    plt.colorbar()
    plt.show()

def plot_phase_resonance_field(i, h, size=128, title="Phase Resonance Field"):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = i * np.sin(np.pi * X * Y * h)

    plt.figure(figsize=(6, 5))
    plt.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='inferno', aspect='auto')
    plt.colorbar(label="Resonance")
    plt.title(title)
    plt.xlabel("X Phase")
    plt.ylabel("Y Phase")
    plt.tight_layout()
    plt.show()

def plot_3d_phase_error_map(i, h, size=64, title="3D Phase Error Map"):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = i * np.sin(np.pi * X * Y * h)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')

    ax.set_title(title)
    ax.set_xlabel("X Phase")
    ax.set_ylabel("Y Phase")
    ax.set_zlabel("Resonance")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()
