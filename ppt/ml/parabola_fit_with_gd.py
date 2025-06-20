import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. 生成数据（真实函数 + 噪声）
np.random.seed(42)
true_a, true_b, true_c = 2.0, -3.0, 5.0
n_points = 100
X = np.linspace(-5, 5, n_points)
noise = np.random.normal(0, 3, size=n_points)
Y = true_a * X**2 - true_b * X + true_c + noise

# 2. 初始化拟合参数
a, b, c = 0.0, 0.0, 0.0
lr = 0.001  # 学习率
epochs = 200
loss_history = []

# 3. 梯度下降函数
def compute_gradients(x, y, a, b, c):
    n = len(x)
    y_pred = a * x**2 - b * x + c
    error = y_pred - y

    da = (2/n) * np.sum(error * x**2)
    db = (-2/n) * np.sum(error * x)
    dc = (2/n) * np.sum(error)

    loss = np.mean(error**2)
    return da, db, dc, loss, y_pred

# 4. 用于动画的数据保存
frames = []

for epoch in range(epochs):
    da, db, dc, loss, y_pred = compute_gradients(X, Y, a, b, c)
    a -= lr * da
    b -= lr * db
    c -= lr * dc
    loss_history.append(loss)
    if epoch % 2 == 0:
        frames.append((a, b, c, y_pred.copy()))

# 5. 可视化：动画展示拟合过程
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

def update(frame):
    ax1.clear()
    ax2.clear()

    a, b, c, y_pred = frame
    ax1.set_title("Fitting Curve")
    ax1.scatter(X, Y, color='blue', label='Data')
    ax1.plot(X, y_pred, color='red', label=f'Fit: {a:.2f}x² - {b:.2f}x + {c:.2f}')
    ax1.legend()
    ax1.set_ylim(min(Y)-10, max(Y)+10)

    ax2.set_title("Loss over Iteration")
    ax2.plot(loss_history[:frames.index(frame)*2+1], color='green')
    ax2.set_ylabel("MSE Loss")
    ax2.set_xlabel("Epoch")

ani = FuncAnimation(fig, update, frames=frames, interval=100)
plt.tight_layout()
plt.show()
