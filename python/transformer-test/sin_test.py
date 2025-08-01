import numpy as np
import matplotlib.pyplot as plt

# Parameters
d = 512
i_values = np.arange(0, 512)
pos_values = [0, 1, 10, 50, 100, 500]

# Create figure with 4 subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, pos in enumerate(pos_values):
    # Calculate the function: sin(pos / (10000^(2i/d)))
    denominator = 10000 ** (2 * i_values / d)
    y_values = np.sin(pos / denominator)
    
    # Plot on the corresponding subplot
    axes[idx].plot(i_values, y_values, linewidth=1.5)
    axes[idx].set_title(f'sin(pos/(10000^(2i/d))) for pos={pos}, d={d}')
    axes[idx].set_xlabel('i')
    axes[idx].set_ylabel('sin value')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
