import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load logs
logs = torch.load("logs_epoch_phase_metrics.pt")

# Settings for the 3D field
bins = 40
field = np.zeros((bins, bins, bins))
arch_sim_range = (-1.0, 1.0)
phase_dist_range = (0.0, 1.0)
rcl_range = (0.0, 1.0)

# Fill the 3D field based on logs
for log in logs:
    i = int((log['archsim'] - arch_sim_range[0]) / (arch_sim_range[1] - arch_sim_range[0]) * (bins - 1))
    j = int((log['phasedist'] - phase_dist_range[0]) / (phase_dist_range[1] - phase_dist_range[0]) * (bins - 1))
    k = int((log['rcl'] - rcl_range[0]) / (rcl_range[1] - rcl_range[0]) * (bins - 1))
    if 0 <= i < bins and 0 <= j < bins and 0 <= k < bins:
        field[i, j, k] += 1

# Visualize the 3D field
x, y, z = np.indices((bins, bins, bins))
mask = field > 0
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x[mask], y[mask], z[mask], c=field[mask], cmap='plasma', s=20)

ax.set_title("Phase Resonance Field")
ax.set_xlabel("ArchSim")
ax.set_ylabel("PhaseDist")
ax.set_zlabel("RCL")
fig.colorbar(sc, label='Resonance Intensity')
plt.tight_layout()
plt.show()