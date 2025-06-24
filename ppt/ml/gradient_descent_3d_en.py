import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Generate simple quadratic function data
np.random.seed(42)
X = np.linspace(-2, 2, 50)
Y = 3 * X**2 + 2 * X + 1 + np.random.normal(0, 0.5, len(X))

# Loss function: for y = ax^2 + bx + c, optimize only a and b (fix c=1)
def loss_function(a, b):
    y_pred = a * X**2 + b * X + 1
    return np.mean((y_pred - Y)**2)

# Compute gradients
def compute_gradients(a, b):
    y_pred = a * X**2 + b * X + 1
    error = y_pred - Y
    da = 2 * np.mean(error * X**2)
    db = 2 * np.mean(error * X)
    return da, db

# Create parameter grid for loss surface
a_range = np.linspace(1, 5, 50)
b_range = np.linspace(0, 4, 50)
A, B = np.meshgrid(a_range, b_range)
Z = np.array([[loss_function(a, b) for a in a_range] for b in b_range])

# Gradient descent
a, b = 1.0, 0.0  # Initial parameters
lr = 0.01
path = [(a, b, loss_function(a, b))]

for _ in range(100):
    da, db = compute_gradients(a, b)
    a -= lr * da
    b -= lr * db
    path.append((a, b, loss_function(a, b)))

path = np.array(path)

# 3D Visualization
fig = plt.figure(figsize=(12, 5))

# Static 3D plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(A, B, Z, alpha=0.6, cmap='viridis')
ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'ro-', markersize=3, linewidth=2)
ax1.set_xlabel('Parameter a')
ax1.set_ylabel('Parameter b')
ax1.set_zlabel('Loss Value')
ax1.set_title('3D Gradient Descent Path')

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(A, B, Z, levels=20)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(path[:, 0], path[:, 1], 'ro-', markersize=4, linewidth=2)
ax2.set_xlabel('Parameter a')
ax2.set_ylabel('Parameter b')
ax2.set_title('Contour View')

plt.tight_layout()
plt.show()

# Animated version
fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection='3d')

def animate(frame):
    ax.clear()
    ax.plot_surface(A, B, Z, alpha=0.3, cmap='viridis')
    
    # Show path up to current frame
    current_path = path[:frame+1]
    ax.plot(current_path[:, 0], current_path[:, 1], current_path[:, 2], 
            'ro-', markersize=4, linewidth=2)
    
    # Current point
    if frame < len(path):
        ax.scatter(path[frame, 0], path[frame, 1], path[frame, 2], 
                  color='red', s=100, alpha=1)
    
    ax.set_xlabel('Parameter a')
    ax.set_ylabel('Parameter b')
    ax.set_zlabel('Loss Value')
    ax.set_title(f'3D Gradient Descent Animation - Step {frame}')

ani = FuncAnimation(fig2, animate, frames=len(path), interval=200, repeat=True)
plt.show()

print(f"Start: a={path[0,0]:.3f}, b={path[0,1]:.3f}, loss={path[0,2]:.3f}")
print(f"End: a={path[-1,0]:.3f}, b={path[-1,1]:.3f}, loss={path[-1,2]:.3f}")
print(f"True values: a=3.0, b=2.0")