import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simple bowl-shaped function: f(x,y) = x² + y²
def f(x, y):
    return x**2 + y**2

# Gradient: ∇f = (2x, 2y)
def gradient(x, y):
    return 2*x, 2*y

# Create meshgrid for surface
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Gradient descent from different starting points
def gradient_descent(start_x, start_y, lr=0.1, steps=20):
    path = [(start_x, start_y)]
    x, y = start_x, start_y
    
    for _ in range(steps):
        dx, dy = gradient(x, y)
        x -= lr * dx
        y -= lr * dy
        path.append((x, y))
    
    return np.array(path)

# Multiple starting points
starts = [(2.5, 2.0), (-2.0, 1.5), (1.0, -2.5), (-1.5, -1.0)]
colors = ['red', 'blue', 'green', 'orange']

# Create visualization
fig = plt.figure(figsize=(15, 5))

# 1. 3D Surface with multiple paths
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')

for i, (sx, sy) in enumerate(starts):
    path = gradient_descent(sx, sy)
    z_path = [f(p[0], p[1]) for p in path]
    ax1.plot(path[:, 0], path[:, 1], z_path, 'o-', 
             color=colors[i], markersize=4, linewidth=2, label=f'Path {i+1}')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(x,y) = x² + y²')
ax1.set_title('3D Gradient Descent Paths')
ax1.legend()

# 2. Contour view
ax2 = fig.add_subplot(132)
contour = ax2.contour(X, Y, Z, levels=15, alpha=0.6)
ax2.clabel(contour, inline=True, fontsize=8)

for i, (sx, sy) in enumerate(starts):
    path = gradient_descent(sx, sy)
    ax2.plot(path[:, 0], path[:, 1], 'o-', 
             color=colors[i], markersize=4, linewidth=2, label=f'Path {i+1}')
    ax2.plot(sx, sy, 's', color=colors[i], markersize=8)  # Start point

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour View')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Loss convergence
ax3 = fig.add_subplot(133)
for i, (sx, sy) in enumerate(starts):
    path = gradient_descent(sx, sy)
    losses = [f(p[0], p[1]) for p in path]
    ax3.plot(losses, 'o-', color=colors[i], linewidth=2, label=f'Path {i+1}')

ax3.set_xlabel('Iteration')
ax3.set_ylabel('Loss Value')
ax3.set_title('Loss Convergence')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print results
print("Gradient Descent Results:")
print("=" * 40)
for i, (sx, sy) in enumerate(starts):
    path = gradient_descent(sx, sy)
    print(f"Path {i+1}: Start({sx:4.1f}, {sy:4.1f}) → End({path[-1,0]:6.3f}, {path[-1,1]:6.3f})")
    print(f"         Loss: {f(sx, sy):6.3f} → {f(path[-1,0], path[-1,1]):6.3f}")
    print()

print("All paths converge to the global minimum at (0, 0) with loss = 0")