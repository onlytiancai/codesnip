"""用 MLX 实现 2 层 MLP 解决鸢尾花 (Iris) 分类

架构: 4 -> 16 -> 3, ReLU 隐藏层, Softmax 输出
损失: 多类交叉熵
优化: 随机梯度下降 (SGD) + 手写反向传播
对比: scikit-learn MLPClassifier (验证实现正确性)
可视化: 训练曲线 + 决策边界

运行: python 017-iris-mlp.py
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


# ---------- 中文字体配置 ----------
def setup_chinese_font() -> None:
    """matplotlib 中文用 PingFang SC, 数学公式用 CM 字体"""
    mpl.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["mathtext.fontset"] = "cm"


# ---------- 数据加载 ----------
def load_data(test_size: float = 0.2, seed: int = 42):
    """加载鸢尾花数据, 标准化, one-hot 编码标签, 80/20 切分"""
    iris = load_iris()
    X, y = iris.data.astype(np.float32), iris.target.astype(np.int32)
    feature_names = iris.feature_names
    target_names = iris.target_names

    # 标准化 (对 MLP 很关键)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # one-hot 编码标签 (n_samples, 3)
    y_train_oh = np.eye(3, dtype=np.float32)[y_train]
    y_test_oh = np.eye(3, dtype=np.float32)[y_test]

    print(f"  样本总数:        {len(X)}")
    print(f"  训练集 / 测试集: {len(X_train)} / {len(X_test)}")
    print(f"  特征:            {feature_names}")
    print(f"  类别:            {list(target_names)}")
    return (X_train, y_train, y_train_oh,
            X_test, y_test, y_test_oh,
            feature_names, target_names)


# ---------- MLX MLP 模型 ----------
class MLP:
    """2 层感知机: Linear -> ReLU -> Linear -> Softmax

    参数量: 4*16 + 16 + 16*3 + 3 = 131
    """

    def __init__(self, in_dim: int = 4, hidden: int = 16, out_dim: int = 3,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # He 初始化 (适合 ReLU)
        self.W1 = mx.array(rng.standard_normal((in_dim, hidden)).astype(np.float32) *
                           np.sqrt(2.0 / in_dim))
        self.b1 = mx.zeros((hidden,))
        self.W2 = mx.array(rng.standard_normal((hidden, out_dim)).astype(np.float32) *
                           np.sqrt(2.0 / in_dim))
        self.b2 = mx.zeros((out_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """前向传播, 返回 softmax 概率"""
        h = mx.maximum(mx.matmul(x, self.W1) + self.b1, 0.0)  # ReLU
        logits = mx.matmul(h, self.W2) + self.b2
        return mx.softmax(logits, axis=-1)

    def forward_with_hidden(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """返回 (h, probs), 反向传播时需要 h"""
        h = mx.maximum(mx.matmul(x, self.W1) + self.b1, 0.0)
        logits = mx.matmul(h, self.W2) + self.b2
        return h, mx.softmax(logits, axis=-1)


def cross_entropy(probs: mx.array, y_true: mx.array) -> mx.array:
    """多类交叉熵 (输入是 softmax 后的概率)"""
    return -mx.mean(mx.sum(y_true * mx.log(probs + 1e-8), axis=-1))


def train_step(model: MLP, x: mx.array, y: mx.array, lr: float
               ) -> tuple[mx.array, mx.array]:
    """一次 SGD 更新, 返回 (loss, probs)"""
    # 前向
    h, probs = model.forward_with_hidden(x)
    loss = cross_entropy(probs, y)

    # 反向 (手写)
    # 1) softmax + cross_entropy 的组合梯度: d_logits = (probs - y) / N
    n = x.shape[0]
    d_logits = (probs - y) / n                        # (N, 3)

    # 2) 第二层: out = h @ W2 + b2
    d_W2 = mx.matmul(h.T, d_logits)                   # (16, 3)
    d_b2 = mx.sum(d_logits, axis=0)                   # (3,)
    d_h = mx.matmul(d_logits, model.W2.T)             # (N, 16)

    # 3) ReLU 梯度: d_pre_relu = d_h * (h > 0)
    d_pre = d_h * (h > 0)

    # 4) 第一层: pre = x @ W1 + b1
    d_W1 = mx.matmul(x.T, d_pre)                      # (4, 16)
    d_b1 = mx.sum(d_pre, axis=0)                      # (16,)

    # 5) SGD 更新
    model.W1 = model.W1 - lr * d_W1
    model.b1 = model.b1 - lr * d_b1
    model.W2 = model.W2 - lr * d_W2
    model.b2 = model.b2 - lr * d_b2

    return loss, probs


def predict(model: MLP, x: mx.array) -> mx.array:
    """返回预测类别索引"""
    probs = model(x)
    return mx.argmax(probs, axis=-1)


def accuracy(model: MLP, x_mx: mx.array, y_np: np.ndarray) -> float:
    """计算分类准确率"""
    pred = predict(model, x_mx)
    return float(mx.mean(pred == mx.array(y_np)).item())


# ---------- sklearn 基准 ----------
def train_sklearn(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    """用 sklearn 的 MLPClassifier 做基准对比"""
    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.05,
        max_iter=200,
        random_state=0,
        batch_size=len(X_train),  # 全量批梯度, 与 MLX 实现一致
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


# ---------- 训练主循环 ----------
def train_model(X_train_oh: np.ndarray, y_train_oh: np.ndarray,
                X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                epochs: int = 200, lr: float = 0.05) -> tuple[MLP, list, list]:
    model = MLP(in_dim=4, hidden=16, out_dim=3, seed=0)
    x_train = mx.array(X_train)
    y_train_mx = mx.array(y_train_oh)
    x_test = mx.array(X_test)

    history = {"loss": [], "train_acc": [], "test_acc": []}

    print(f"\n  开始训练 (MLP 4→16→3, lr={lr}, epochs={epochs})")
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Train Acc':>10}  {'Test Acc':>10}")
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss, _ = train_step(model, x_train, y_train_mx, lr)
        mx.eval(loss)  # 强制求值, 才能拿到标量
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            tr = accuracy(model, x_train, y_train)
            te = accuracy(model, x_test, y_test)
            history["loss"].append((epoch, loss.item()))
            history["train_acc"].append((epoch, tr))
            history["test_acc"].append((epoch, te))
            print(f"  {epoch:>6}  {loss.item():>10.4f}  {tr * 100:>9.2f}%  {te * 100:>9.2f}%")
    train_ms = (time.perf_counter() - t0) * 1000
    print(f"  训练耗时: {train_ms:.1f} ms")

    # 最终一次评估
    final_train = accuracy(model, x_train, y_train)
    final_test = accuracy(model, x_test, y_test)
    history["final_train_acc"] = final_train
    history["final_test_acc"] = final_test
    return model, history, history


# ---------- 可视化 ----------
def plot_curves(history: dict, save_path: str = "/tmp/iris_training.png") -> None:
    """训练曲线: loss + accuracy"""
    eps_l, losses = zip(*history["loss"])
    _, train_acc = zip(*history["train_acc"])
    _, test_acc = zip(*history["test_acc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(eps_l, losses, "o-", color="#3a7bd5", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("训练损失曲线")
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

    fig.suptitle("MLX MLP 鸢尾花分类训练曲线", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    print(f"\n  训练曲线已保存到 {save_path}")


def plot_decision_boundary(model: MLP, X: np.ndarray, y: np.ndarray,
                           target_names: list[str],
                           feature_names: list[str],
                           save_path: str = "/tmp/iris_boundary.png") -> None:
    """用前两个特征 (花萼长 / 花萼宽) 画决策边界"""
    # 只用前 2 个特征: 需要重训一个 2 维输入的模型用于可视化
    X2 = X[:, :2]
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y, test_size=0.2, random_state=42, stratify=y
    )
    model2 = MLP(in_dim=2, hidden=16, out_dim=3, seed=0)
    y2_train_oh = np.eye(3, dtype=np.float32)[y2_train]
    x2_tr = mx.array(X2_train)
    y2_tr_mx = mx.array(y2_train_oh)
    for _ in range(300):
        train_step(model2, x2_tr, y2_tr_mx, lr=0.05)

    # 构造网格
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = mx.array(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    probs = np.array(model2(grid))
    pred = probs.argmax(axis=-1).reshape(xx.shape)

    colors = ["#ff9999", "#99ff99", "#9999ff"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, pred, alpha=0.3, cmap=mpl.colors.ListedColormap(colors))
    for i, name in enumerate(target_names):
        mask = y == i
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=colors[i], edgecolor="k", s=40, label=name)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("决策边界 (使用前 2 个特征)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    print(f"  决策边界已保存到 {save_path}")


# ---------- 主入口 ----------
def main() -> None:
    setup_chinese_font()
    print("=" * 60)
    print("  MLX MLP 鸢尾花分类")
    print("=" * 60)

    # 1. 数据
    (X_train, y_train, y_train_oh,
     X_test, y_test, y_test_oh,
     feature_names, target_names) = load_data()

    # 2. MLX 训练
    print("\n" + "=" * 60)
    print("  MLX 实现")
    print("=" * 60)
    model, history, _ = train_model(
        X_train, y_train_oh, X_train, y_train, X_test, y_test,
        epochs=200, lr=0.05,
    )
    print(f"\n  最终结果:")
    print(f"    训练准确率: {history['final_train_acc'] * 100:.2f}%")
    print(f"    测试准确率: {history['final_test_acc'] * 100:.2f}%")

    # 3. sklearn 基准
    print("\n" + "=" * 60)
    print("  sklearn 基准对比")
    print("=" * 60)
    sk_train, sk_test = train_sklearn(X_train, y_train, X_test, y_test)
    print(f"\n  对比小结:")
    print(f"    MLX 测试准确率:      {history['final_test_acc'] * 100:.2f}%")
    print(f"    sklearn 测试准确率:  {sk_test * 100:.2f}%")
    diff = abs(history['final_test_acc'] - sk_test)
    if diff < 0.05:
        print(f"    ✅ 差距 {diff * 100:.2f}%, 实现正确")
    else:
        print(f"    ⚠️  差距 {diff * 100:.2f}%, 可能有问题")

    # 4. 可视化
    print("\n" + "=" * 60)
    print("  生成可视化图表")
    print("=" * 60)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    plot_curves(history)
    plot_decision_boundary(model, X_all, y_all, list(target_names), feature_names)
    print("\n✨ 全部完成\n")


if __name__ == "__main__":
    main()