import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define different loss functions for demonstration
def simple_bowl(x, y):
    """Simple quadratic bowl: f(x,y) = x² + y²"""
    return x**2 + y**2

def saddle_point(x, y):
    """Saddle point: f(x,y) = x² - y²"""
    return x**2 - y**2

def rosenbrock(x, y):
    """Rosenbrock function (scaled down)"""
    return 0.1 * ((1 - x)**2 + 10 * (y - x**2)**2)

# Corresponding gradients
def grad_bowl(x, y):
    return 2*x, 2*y

def grad_saddle(x, y):
    return 2*x, -2*y

def grad_rosenbrock(x, y):
    dx = 0.1 * (-2 * (1 - x) - 40 * x * (y - x**2))
    dy = 0.1 * (20 * (y - x**2))
    return dx, dy

# Gradient descent algorithm
def gradient_descent(grad_func, start, lr=0.1, steps=50):
    path = [start]
    x, y = start
    
    for _ in range(steps):
        dx, dy = grad_func(x, y)
        x -= lr * dx
        y -= lr * dy
        path.append((x, y))
    
    return np.array(path)

# Choose function to visualize
func = simple_bowl
grad_func = grad_bowl
func_name = "Bowl Function: f(x,y) = x² + y²"

# Create meshgrid
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

# Multiple starting points
starts = [(2.5, 2.0), (-2.0, 1.5), (1.0, -2.5)]
colors = ['red', 'blue', 'green']

# Generate paths
paths = []
for start in starts:
    path = gradient_descent(grad_func, start)
    paths.append(path)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. 3D Surface with paths
ax1 = fig.add_subplot(231, projection='3d')
surf = ax1.plot_surface(X, Y, Z, alpha=0.4, cmap='viridis')
for i, path in enumerate(paths):
    z_path = [func(p[0], p[1]) for p in path]
    ax1.plot(path[:, 0], path[:, 1], z_path, 'o-', 
             color=colors[i], markersize=3, linewidth=2)
ax1.set_title('3D Surface View')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(x,y)')

# 2. Contour with paths
ax2 = fig.add_subplot(232)
contour = ax2.contour(X, Y, Z, levels=20, alpha=0.6)
ax2.clabel(contour, inline=True, fontsize=8)
for i, path in enumerate(paths):
    ax2.plot(path[:, 0], path[:, 1], 'o-', 
             color=colors[i], markersize=3, linewidth=2)
    ax2.plot(path[0, 0], path[0, 1], 's', color=colors[i], markersize=8)
ax2.set_title('Contour View')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(True, alpha=0.3)

# 3. Gradient vector field
ax3 = fig.add_subplot(233)
x_sparse = np.linspace(-3, 3, 15)
y_sparse = np.linspace(-3, 3, 15)
X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)
DX, DY = grad_func(X_sparse, Y_sparse)
ax3.quiver(X_sparse, Y_sparse, -DX, -DY, alpha=0.6, scale=50)
ax3.contour(X, Y, Z, levels=10, alpha=0.3)
for i, path in enumerate(paths):
    ax3.plot(path[:, 0], path[:, 1], 'o-', 
             color=colors[i], markersize=3, linewidth=2)
ax3.set_title('Gradient Vector Field')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')

# 4. Loss convergence
ax4 = fig.add_subplot(234)
for i, path in enumerate(paths):
    losses = [func(p[0], p[1]) for p in path]
    ax4.plot(losses, 'o-', color=colors[i], linewidth=2, label=f'Path {i+1}')
ax4.set_title('Loss Convergence')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Loss Value')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Parameter trajectories
ax5 = fig.add_subplot(235)
for i, path in enumerate(paths):
    ax5.plot(path[:, 0], 'o-', color=colors[i], linewidth=2, label=f'X Path {i+1}')
    ax5.plot(path[:, 1], 's--', color=colors[i], linewidth=2, alpha=0.7, label=f'Y Path {i+1}')
ax5.set_title('Parameter Evolution')
ax5.set_xlabel('Iteration')
ax5.set_ylabel('Parameter Value')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Step sizes
ax6 = fig.add_subplot(236)
for i, path in enumerate(paths):
    step_sizes = []
    for j in range(1, len(path)):
        step = np.linalg.norm(path[j] - path[j-1])
        step_sizes.append(step)
    ax6.plot(step_sizes, 'o-', color=colors[i], linewidth=2, label=f'Path {i+1}')
ax6.set_title('Step Sizes')
ax6.set_xlabel('Iteration')
ax6.set_ylabel('Step Size')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle(func_name, fontsize=16)
plt.tight_layout()
plt.show()

# Create animated version
fig2 = plt.figure(figsize=(12, 8))
ax = fig2.add_subplot(111, projection='3d')

def animate_comprehensive(frame):
    ax.clear()
    
    # Draw surface
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')
    
    # Draw paths up to current frame
    for i, path in enumerate(paths):
        if frame < len(path):
            current_path = path[:frame+1]
            z_path = [func(p[0], p[1]) for p in current_path]
            
            # Path
            if len(current_path) > 1:
                ax.plot(current_path[:, 0], current_path[:, 1], z_path, 
                       'o-', color=colors[i], markersize=3, linewidth=2, alpha=0.8)
            
            # Current point
            ax.scatter(path[frame, 0], path[frame, 1], z_path[-1], 
                      color=colors[i], s=100, alpha=1)
            
            # Gradient arrow
            if frame > 0:
                dx, dy = grad_func(path[frame, 0], path[frame, 1])
                ax.quiver(path[frame, 0], path[frame, 1], z_path[-1],
                         -dx*0.5, -dy*0.5, 0, color=colors[i], alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x,y)')
    ax.set_title(f'3D Gradient Descent Animation - Step {frame}')
    ax.view_init(elev=30, azim=45)

# Create animation
max_frames = max(len(path) for path in paths)
ani = FuncAnimation(fig2, animate_comprehensive, frames=max_frames, 
                   interval=300, repeat=True)
plt.show()

# Print summary
print("3D Gradient Descent Visualization Summary")
print("=" * 50)
print(f"Function: {func_name}")
print(f"Learning Rate: 0.1")
print(f"Max Iterations: 50")
print()

for i, (start, path) in enumerate(zip(starts, paths)):
    end = path[-1]
    start_loss = func(start[0], start[1])
    end_loss = func(end[0], end[1])
    print(f"Path {i+1}:")
    print(f"  Start: ({start[0]:5.1f}, {start[1]:5.1f}) → Loss: {start_loss:8.3f}")
    print(f"  End:   ({end[0]:5.3f}, {end[1]:5.3f}) → Loss: {end_loss:8.6f}")
    print(f"  Reduction: {((start_loss - end_loss) / start_loss * 100):5.1f}%")
    print()