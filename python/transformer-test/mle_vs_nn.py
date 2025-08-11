# 优化版：减少网格密度与训练轮数
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
from torch import nn

# 1. 数据准备（前两维便于可视化）
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. LDA 模型
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 3. 神经网络模型（减少轮数）
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(100):  # 减少训练轮数
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# 4. 绘制决策边界
def plot_decision_boundary(clf, X, y, title, is_nn=False):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if is_nn:
        grid_t = torch.tensor(grid, dtype=torch.float32)
        Z = model(grid_t).detach().numpy()
        Z = np.argmax(Z, axis=1)
    else:
        Z = clf.predict(grid)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Set1)
    plt.title(title)
    plt.xlabel("Feature 1 (std)")
    plt.ylabel("Feature 2 (std)")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plot_decision_boundary(lda, X, y, "LDA (MLE)", is_nn=False)
plt.subplot(1, 2, 2)
plot_decision_boundary(None, X, y, "Neural Network", is_nn=True)
plt.tight_layout()
plt.show()
