import numpy as np

def sigmoid(z):
    # 为数值稳定用 np.clip
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def dsigmoid(a):
    """sigmoid 对 a 本身求导（a 已经是 sigmoid(z)）"""
    return a * (1.0 - a)


class MLP:
    """
    两层感知机：输入(2) -> 隐藏(h) -> 输出(1)
    激活: sigmoid + sigmoid
    损失: 二元交叉熵
    """
    def __init__(self, n_in, n_hidden=4, lr=0.5, seed=0):
        # n_in   : 输入层维度（特征数量）
        # n_hidden: 隐藏层神经元个数
        # lr     : 学习率（梯度下降步长）
        # seed   : 随机种子，保证结果可复现
        rng = np.random.default_rng(seed)
        # Xavier 初始化：以 N(0, 1/√n_in) 采样第一层权重
        #   相比纯 He 初始化，更适配 sigmoid 这类对称激活函数，
        #   有助于避免训练初期梯度消失/爆炸
        self.W1 = rng.normal(0, 1/np.sqrt(n_in),   size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)                # 隐藏层偏置，初始化为 0
        # 第二层权重同样使用 Xavier，方差按上一层的 fan-in (n_hidden) 计算
        self.W2 = rng.normal(0, 1/np.sqrt(n_hidden), size=(n_hidden, 1))
        self.b2 = np.zeros(1)                       # 输出层偏置，初始化为 0
        self.lr = lr                                # 记录学习率，供 backward 使用
        self.loss_history = []                      # 保存每轮的损失值，便于观察收敛曲线

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1            # (N, h)
        self.a1 = sigmoid(self.z1)                 # (N, h)
        self.z2 = self.a1 @ self.W2 + self.b2      # (N, 1)
        self.a2 = sigmoid(self.z2).ravel()         # (N,)
        return self.a2

    def backward(self, X, y, y_pred):
        N = len(X)
        y_pred = y_pred.reshape(-1, 1)             # (N, 1)

        # 输出层
        dz2 = (y_pred - y.reshape(-1, 1))          # (N, 1)  ← 交叉熵+sigmoid 的简化
        dW2 = (self.a1.T @ dz2) / N                # (h, 1)
        db2 = dz2.mean(axis=0)                     # (1,)

        # 隐藏层
        da1 = dz2 @ self.W2.T                      # (N, h)
        dz1 = da1 * dsigmoid(self.a1)              # (N, h)
        dW1 = (X.T @ dz1) / N                      # (2, h)
        db1 = dz1.mean(axis=0)                     # (h,)

        # 梯度下降
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, y, epochs=5000, verbose_every=1000):
        for ep in range(1, epochs + 1):
            y_pred = self.forward(X)
            # 交叉熵损失（加 1e-8 防止 log(0)）
            loss = -np.mean(y * np.log(y_pred + 1e-8) +
                            (1 - y) * np.log(1 - y_pred + 1e-8))
            self.loss_history.append(loss)
            self.backward(X, y, y_pred)
            if verbose_every and (ep % verbose_every == 0 or ep == 1):
                acc = (y_pred.round() == y).mean()
                print(f"epoch {ep:5d}  loss={loss:.4f}  acc={acc:.2f}")
        return self

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y_xor = np.array([0, 1, 1, 0])

# ---------- 在 XOR 上训练 ----------
mlp = MLP(n_in=2, n_hidden=4, lr=1.0, seed=1)
mlp.fit(X, y_xor, epochs=10000, verbose_every=2000)

print("\n最终预测概率:", mlp.forward(X).round(3))
print("最终预测类别:", mlp.predict(X))
print("真实标签    :", y_xor)