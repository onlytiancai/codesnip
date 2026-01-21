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

for sent in sentences:
    words = sent.split()
    for i in range(len(words) - 2):
        w1, w2, w3 = words[i], words[i+1], words[i+2]
        x = np.concatenate([one_hot(w1), one_hot(w2)])
        y = one_hot(w3)
        X_list.append(x)
        Y_list.append(y)

X = np.stack(X_list, axis=1)   # (2V, T)
Y = np.stack(Y_list, axis=1)   # (V, T)

# ---------------------------
# 5. 最小二乘解
# ---------------------------
# W = Y X^T (X X^T)^(-1)
XXT = X @ X.T

# 调试信息：检查矩阵条件
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"XXT shape: {XXT.shape}")
print(f"XXT rank: {np.linalg.matrix_rank(XXT)}")
print(f"XXT condition number: {np.linalg.cond(XXT)}")

# 使用最小二乘法直接求解（更稳定）
# W.T = (X.T)^+ Y.T 或 W = Y @ X^+
# 使用 lstsq 方法对每一行分别求解
W_rows = []
for i in range(Y.shape[0]):
    y_i = Y[i, :]
    # 求解 w_i @ X = y_i，即 X.T @ w_i.T = y_i.T
    w_i, residuals, rank, s = np.linalg.lstsq(X.T, y_i, rcond=None)
    W_rows.append(w_i)

W = np.array(W_rows)
# W = Y @ X.T @ np.linalg.pinv(XXT) # use pinv

# 检查结果
print(f"W shape: {W.shape}")
print(f"W contains NaN: {np.any(np.isnan(W))}")
print(f"W contains Inf: {np.any(np.isinf(W))}")

# ---------------------------
# 6. 预测函数
# ---------------------------
def predict(w1, w2):
    x = np.concatenate([one_hot(w1), one_hot(w2)])
    y_hat = W @ x
    idx = int(np.argmax(y_hat))
    return id2word[idx]

# ---------------------------
# 7. 演示预测
# ---------------------------
tests = [
    ("我", "喜欢"),
    ("喜欢", "学习"),
    ("学习", "是"),
    ("是", "一个"),
    ("一个","方法"),
    ("方法","可以"),    
]

for w1, w2 in tests:
    print(f"{w1} {w2} -> {predict(w1, w2)}")
