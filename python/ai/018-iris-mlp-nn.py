"""用 mlx.nn 重构的鸢尾花 MLP 分类器

对比 017-iris-mlp.py 的手写实现:
- 模型层: 自管 W/b 参数 → nn.Linear (内置 He 风格初始化)
- 激活: 手写 ReLU (max(., 0)) → nn.relu
- 损失: 手写 softmax + cross_entropy → nn.losses.cross_entropy (接受 logits)
- 梯度: 手写反向传播 4 层链式求导 → nn.value_and_grad (自动)
- 优化器: 手写 SGD → optimizers.SGD/Adam

数据加载、可视化、sklearn 基准对比与 017 完全一致, 方便横向对比.

运行: python 018-iris-mlp-nn.py
"""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.nn.losses as losses


# ---------- 中文字体配置 ----------
def setup_chinese_font() -> None:
    mpl.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["mathtext.fontset"] = "cm"


# ---------- 数据加载 ----------
def load_data(test_size: float = 0.2, seed: int = 42):
    iris = load_iris()
    X, y = iris.data.astype(np.float32), iris.target.astype(np.int32)
    feature_names = iris.feature_names
    target_names = iris.target_names

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"  样本总数:        {len(X)}")
    print(f"  训练集 / 测试集: {len(X_train)} / {len(X_test)}")
    print(f"  特征:            {feature_names}")
    print(f"  类别:            {list(target_names)}")
    return X_train, y_train, X_test, y_test, feature_names, target_names


# ---------- nn.Module 模型 ----------
class MLP(nn.Module):
    """2 层感知机: Linear -> ReLU -> Linear -> Softmax

    参数量: 4*16 + 16 + 16*3 + 3 = 131
    注意 nn.Linear 默认 weight shape 为 (out, in), y = x @ W.T + b,
    MLX 在 forward 时会自动处理转置.
    """

    def __init__(self, in_dim: int = 4, hidden: int = 16, out_dim: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """返回 logits (未 softmax), 配合 nn.losses.cross_entropy 使用"""
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


# ---------- 训练 ----------
def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                epochs: int = 300, lr: float = 0.1) -> tuple[MLP, dict]:
    model = MLP(in_dim=4, hidden=16, out_dim=3)
    # 训练时不评估模式 (虽然本例没 dropout/batchnorm, 但保持规范)
    model.train()

    # 标签转 MLX 数组 (交叉熵接受整数索引, 不需要 one-hot)
    y_train_mx = mx.array(y_train)
    y_test_mx = mx.array(y_test)
    x_train_mx = mx.array(X_train)
    x_test_mx = mx.array(X_test)

    # 优化器 (0.32+ 内部自动管理 state, 无需 init) — Adam 自适应学习率, 收敛更快
    optimizer = optim.Adam(learning_rate=lr)

    # 损失函数: nn.value_and_grad 的 fn 只接受 model, x/y 用闭包捕获
    def loss_fn(model_to_grad):
        logits = model_to_grad(x_train_mx)
        return losses.cross_entropy(logits, y_train_mx, reduction="mean")

    grad_fn = nn.value_and_grad(model, loss_fn)

    def eval_acc(x, y) -> float:
        logits = model(x)
        pred = mx.argmax(logits, axis=-1)
        return float(mx.mean(pred == y).item())

    history = {"loss": [], "train_acc": [], "test_acc": []}

    print(f"\n  开始训练 (MLP 4→16→3, Adam lr={lr}, epochs={epochs})")
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Train Acc':>10}  {'Test Acc':>10}")

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # 前向 + 反向
        loss, grads = grad_fn(model)
        # 更新参数 (0.32+ 自动管理 state)
        optimizer.update(model, grads)
        # MLX 是懒计算, 强制求值, 让 loss 在下一行立即拿到标量
        mx.eval(model.parameters(), loss)

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            l_val = loss.item()
            tr = eval_acc(x_train_mx, y_train_mx)
            te = eval_acc(x_test_mx, y_test_mx)
            history["loss"].append((epoch, l_val))
            history["train_acc"].append((epoch, tr))
            history["test_acc"].append((epoch, te))
            print(f"  {epoch:>6}  {l_val:>10.4f}  {tr * 100:>9.2f}%  {te * 100:>9.2f}%")
    train_ms = (time.perf_counter() - t0) * 1000
    print(f"  训练耗时: {train_ms:.1f} ms")

    # 最终评估
    history["final_train_acc"] = eval_acc(x_train_mx, y_train_mx)
    history["final_test_acc"] = eval_acc(x_test_mx, y_test_mx)
    return model, history


# ---------- sklearn 基准 (与 017 一致) ----------
def train_sklearn(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.05,
        max_iter=200,
        random_state=0,
        batch_size=len(X_train),
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_ms = (time.perf_counter() - t0) * 1000
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n  sklearn MLPClassifier:")
    print(f"    训练耗时:    {train_ms:.1f} ms (200 epochs)")
    print(f"    训练准确率:  {train_acc * 100:.1f}%")
    print(f"    测试准确率:  {test_acc * 100:.1f}%")
    return train_acc, test_acc


# ---------- 可视化 ----------
def plot_curves(history: dict, save_path: str = "/tmp/iris_nn_training.png") -> None:
    eps_l, losses_v = zip(*history["loss"])
    _, train_acc = zip(*history["train_acc"])
    _, test_acc = zip(*history["test_acc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(eps_l, losses_v, "o-", color="#3a7bd5", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("训练损失曲线 (mlx.nn)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps_l, [a * 100 for a in train_acc], "o-",
             color="#2ecc71", linewidth=2, label="训练集")
    ax2.plot(eps_l, [a * 100 for a in test_acc], "s--",
             color="#e74c3c", linewidth=2, label="测试集")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("准确率 (%)")
    ax2.set_title("分类准确率")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("MLX nn.Module 鸢尾花分类训练曲线", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    print(f"\n  训练曲线已保存到 {save_path}")


def plot_decision_boundary(X: np.ndarray, y: np.ndarray,
                           target_names: list[str], feature_names: list[str],
                           save_path: str = "/tmp/iris_nn_boundary.png") -> None:
    """用前 2 个特征 (花萼长 / 花萼宽) 画决策边界"""
    X2 = X[:, :2]
    X2_train, _, y2_train, _ = train_test_split(
        X2, y, test_size=0.2, random_state=42, stratify=y
    )

    # 重新训一个 2 维输入的模型用于可视化
    model2 = MLP(in_dim=2, hidden=16, out_dim=3)
    model2.train()
    optimizer = optim.Adam(learning_rate=0.1)
    x_mx = mx.array(X2_train)
    y_mx = mx.array(y2_train)

    def loss_fn2(model_to_grad):
        return losses.cross_entropy(
            model_to_grad(x_mx), y_mx, reduction="mean"
        )

    grad_fn2 = nn.value_and_grad(model2, loss_fn2)
    for _ in range(300):
        loss, grads = grad_fn2(model2)
        optimizer.update(model2, grads)
        mx.eval(model2.parameters(), loss)
    model2.eval()

    # 构造网格
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = mx.array(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    logits = model2(grid)
    probs = mx.softmax(logits, axis=-1)
    pred = np.array(mx.argmax(probs, axis=-1)).reshape(xx.shape)

    colors = ["#ff9999", "#99ff99", "#9999ff"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, pred, alpha=0.3, cmap=mpl.colors.ListedColormap(colors))
    for i, name in enumerate(target_names):
        mask = y == i
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=colors[i], edgecolor="k", s=40, label=name)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("决策边界 (mlx.nn, 前 2 个特征)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    print(f"  决策边界已保存到 {save_path}")


# ---------- 主入口 ----------
def main() -> None:
    setup_chinese_font()
    print("=" * 60)
    print("  MLX nn.Module 鸢尾花分类 (重构版)")
    print("=" * 60)

    # 1. 数据
    X_train, y_train, X_test, y_test, feature_names, target_names = load_data()

    # 2. MLX nn 训练
    print("\n" + "=" * 60)
    print("  MLX nn.Module 实现")
    print("=" * 60)
    model, history = train_model(X_train, y_train, X_test, y_test,
                                 epochs=300, lr=0.1)
    print(f"\n  最终结果:")
    print(f"    训练准确率: {history['final_train_acc'] * 100:.2f}%")
    print(f"    测试准确率: {history['final_test_acc'] * 100:.2f}%")

    # 3. sklearn 基准
    print("\n" + "=" * 60)
    print("  sklearn 基准对比")
    print("=" * 60)
    sk_train, sk_test = train_sklearn(X_train, y_train, X_test, y_test)
    print(f"\n  对比小结:")
    print(f"    MLX nn 测试准确率:    {history['final_test_acc'] * 100:.2f}%")
    print(f"    sklearn 测试准确率:   {sk_test * 100:.2f}%")
    diff = abs(history['final_test_acc'] - sk_test)
    if diff < 0.05:
        print(f"    ✅ 差距 {diff * 100:.2f}%, 接近 sklearn 水平")
    else:
        print(f"    ⚠️  差距 {diff * 100:.2f}%")

    # 4. 可视化
    print("\n" + "=" * 60)
    print("  生成可视化图表")
    print("=" * 60)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    plot_curves(history)
    plot_decision_boundary(X_all, y_all, list(target_names), feature_names)

    # 5. 模型信息
    print("\n" + "=" * 60)
    print("  模型信息")
    print("=" * 60)
    print(f"  结构:\n{model}")
    # parameters() 是嵌套 dict {layer: {param_name: array}}, 展平统计
    n_params = sum(
        p.size for layer_params in model.parameters().values()
        for p in layer_params.values()
    )
    print(f"\n  参数量: {n_params}")
    print("\n✨ 全部完成\n")


if __name__ == "__main__":
    main()