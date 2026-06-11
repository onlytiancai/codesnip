import numpy as np

# ---------------------------
# 1. 词表
# ---------------------------
vocab = [
    "我", "你", "他", "我们", "他们",
    "是", "不是", "有", "没有",
    "喜欢", "学习", "工作", "生活",
    "机器", "模型", "方法",
    "可以", "用", "来", "做",
    "一个", "这个", "那个"
]

word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}
V = len(vocab)

# ---------------------------
# 2. 训练语料
# ---------------------------
sentences = [
    "我 喜欢 学习",
    "我 喜欢 工作",
    "你 喜欢 学习",
    "他 喜欢 工作",
    "我们 喜欢 生活",
    "他们 喜欢 学习",

    "机器 学习 是 一个 方法",
    "机器 学习 是 一个 模型",
    "这个 模型 可以 用 来 学习",
    "这个 方法 可以 用 来 工作",

    "我 用 机器 学习",
    "我们 用 机器 学习",
    "他们 用 一个 模型",
]

# ---------------------------
# 3. one-hot 编码
# ---------------------------
def one_hot(word):
    v = np.zeros(V)
    v[word2id[word]] = 1.0
    return v

# ---------------------------
# 4. 构造 2-gram 样本
# ---------------------------
X_list = []
Y_list = []
Y_indices = []

for sent in sentences:
    words = sent.split()
    for i in range(len(words) - 2):
        w1, w2, w3 = words[i], words[i+1], words[i+2]
        x = np.concatenate([one_hot(w1), one_hot(w2)])
        y = one_hot(w3)
        X_list.append(x)
        Y_list.append(y)
        Y_indices.append(word2id[w3])

X = np.stack(X_list, axis=0)   # (T, 2V)
Y = np.stack(Y_list, axis=0)   # (T, V)
Y_idx = np.array(Y_indices)    # (T,)

T = X.shape[0]
print(f"Dataset: {T} samples, Input dim: {2*V}, Output dim: {V}")
print("=" * 70)

# ============================================================================
# 方法 1: 最小二乘法 (Least Squares)
# ============================================================================
print("\n方法 1: 最小二乘法 (Least Squares)")
print("-" * 70)

# 转换为矩阵形式用于最小二乘
# Y = W X^T，但这里的X是(T, 2V)，所以需要转置
# 最小二乘：W = Y X^T (X X^T)^(-1)
X_T = X.T  # (2V, T)
Y_T = Y.T  # (V, T)

XXT = X_T @ X_T.T  # (2V, 2V)

print(f"XXT shape: {XXT.shape}")
print(f"XXT rank: {np.linalg.matrix_rank(XXT)}")
print(f"XXT condition number: {np.linalg.cond(XXT):.2e}")

W_ls = Y_T @ X_T.T @ np.linalg.pinv(XXT)  # (V, 2V)

print(f"W_ls shape: {W_ls.shape}")
print(f"W_ls contains NaN: {np.any(np.isnan(W_ls))}")
print(f"W_ls contains Inf: {np.any(np.isinf(W_ls))}")

def predict_ls(w1, w2):
    """最小二乘模型预测"""
    x = np.concatenate([one_hot(w1), one_hot(w2)])
    y_hat = W_ls @ x
    idx = int(np.argmax(y_hat))
    return id2word[idx], y_hat[idx]

# 最小二乘法的预测
print("\n最小二乘法 - 预测结果:")
tests = [
    ("我", "喜欢"),
    ("喜欢", "学习"),
    ("学习", "是"),
    ("是", "一个"),
    ("一个","方法"),
    ("方法","可以"),    
]

for w1, w2 in tests:
    pred, score = predict_ls(w1, w2)
    print(f"  {w1} {w2} -> {pred:6s} (score: {score:7.4f})")

# 最小二乘法的准确率
y_pred_ls = X @ W_ls.T  # (T, V)
pred_idx_ls = np.argmax(y_pred_ls, axis=1)
acc_ls = np.mean(pred_idx_ls == Y_idx)
print(f"\n最小二乘法 - 训练集准确率: {acc_ls:.4f}")

# ============================================================================
# 方法 2: Softmax + 交叉熵损失 (Softmax + Cross Entropy)
# ============================================================================
print("\n" + "=" * 70)
print("方法 2: Softmax + 交叉熵损失 (Softmax + Cross Entropy)")
print("-" * 70)

# 初始化参数
np.random.seed(42)
W_softmax = np.random.randn(2*V, V) * 0.01
b_softmax = np.zeros(V)

# 超参数
learning_rate = 0.1
num_epochs = 200

def softmax(z):
    """数值稳定的softmax"""
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """交叉熵损失"""
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss

# 训练
print(f"\n训练配置: learning_rate={learning_rate}, epochs={num_epochs}")
print(f"参数: W shape {W_softmax.shape}, b shape {b_softmax.shape}\n")

losses = []

for epoch in range(num_epochs):
    # 前向传播
    logits = X @ W_softmax + b_softmax  # (T, V)
    probs = softmax(logits)              # (T, V)
    
    # 计算损失
    loss = cross_entropy_loss(Y, probs)
    losses.append(loss)
    
    # 反向传播
    dlogits = probs - Y  # (T, V)
    dW = X.T @ dlogits  # (2V, V)
    db = np.sum(dlogits, axis=0)  # (V,)
    
    # 更新参数
    W_softmax -= learning_rate * dW
    b_softmax -= learning_rate * db
    
    if (epoch + 1) % 50 == 0:
        pred_idx_ce = np.argmax(probs, axis=1)
        acc_ce = np.mean(pred_idx_ce == Y_idx)
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {loss:.6f} | Train Acc: {acc_ce:.4f}")

def predict_softmax(w1, w2):
    """Softmax + 交叉熵模型预测"""
    x = np.concatenate([one_hot(w1), one_hot(w2)])
    logits = x @ W_softmax + b_softmax
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return id2word[idx], probs[idx]

# Softmax预测
print("\nSoftmax + 交叉熵 - 预测结果:")
for w1, w2 in tests:
    pred, conf = predict_softmax(w1, w2)
    print(f"  {w1} {w2} -> {pred:6s} (confidence: {conf:.4f})")

# Softmax准确率
logits = X @ W_softmax + b_softmax
probs = softmax(logits)
pred_idx_ce = np.argmax(probs, axis=1)
acc_ce = np.mean(pred_idx_ce == Y_idx)
print(f"\nSoftmax + 交叉熵 - 训练集准确率: {acc_ce:.4f}")

# ============================================================================
# 3. 方法对比
# ============================================================================
print("\n" + "=" * 70)
print("方法对比总结")
print("=" * 70)

print("\n方法 1: 最小二乘法 (Least Squares)")
print(f"  - 参数数量: {W_ls.size}")
print(f"  - 训练时间: O(n³) 矩阵求逆")
print(f"  - 训练集准确率: {acc_ls:.4f}")
print(f"  - 优点: 直接求解，无需迭代")
print(f"  - 缺点: 输出不是概率，无法直接应用概率损失")

print("\n方法 2: Softmax + 交叉熵 (Softmax + Cross Entropy)")
print(f"  - 参数数量: {W_softmax.size + b_softmax.size}")
print(f"  - 训练时间: 梯度下降迭代")
print(f"  - 训练集准确率: {acc_ce:.4f}")
print(f"  - 优点: 输出为概率分布，理论基础清晰，扩展性好")
print(f"  - 缺点: 需要迭代，需要设置超参数")

print("\n" + "=" * 70)
