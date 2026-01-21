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
W = Y @ X.T @ np.linalg.pinv(XXT)

# ---------------------------
# 6. 预测函数
# ---------------------------
def predict(w1, w2):
    x = np.concatenate([one_hot(w1), one_hot(w2)])
    y_hat = W @ x
    idx = np.argmax(y_hat)
    return id2word[idx]

# ---------------------------
# 7. 演示预测
# ---------------------------
tests = [
    ("我", "喜欢"),
    ("机器", "学习"),
    ("这个", "模型"),
    ("我们", "用"),
]

for w1, w2 in tests:
    print(f"{w1} {w2} -> {predict(w1, w2)}")
