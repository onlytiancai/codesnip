import numpy as np
import matplotlib.pyplot as plt

# ----- Vocab and data -----
vocab = [
    "我", "你", "他", "我们", "他们",
    "是", "不是", "有", "没有",
    "喜欢", "学习", "工作", "生活",
    "机器", "模型", "方法",
    "可以", "用", "来", "做",
    "一个", "这个", "那个"
]
word2id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

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

# ----- One-hot -----
def one_hot(word):
    v = np.zeros(V)
    v[word2id[word]] = 1.0
    return v

# ----- Build one-hot concatenation X -----
X_onehot = []
for sent in sentences:
    words = sent.split()
    for i in range(len(words) - 2):
        X_onehot.append(
            np.concatenate([one_hot(words[i]), one_hot(words[i+1])])
        )
X_onehot = np.stack(X_onehot, axis=1)   # (2V, T)

# ----- Random embedding -----
d = 8  # embedding dimension
rng = np.random.default_rng(0)
E = rng.standard_normal((V, d)) / np.sqrt(d)

def embed(word):
    return E[word2id[word]]

# ----- Build embedding concatenation X -----
X_emb = []
for sent in sentences:
    words = sent.split()
    for i in range(len(words) - 2):
        X_emb.append(
            np.concatenate([embed(words[i]), embed(words[i+1])])
        )
X_emb = np.stack(X_emb, axis=1)   # (2d, T)

# ----- XX^T -----
XXT_onehot = X_onehot @ X_onehot.T
XXT_emb = X_emb @ X_emb.T

# ----- Eigenvalues -----
eig_onehot = np.sort(np.linalg.eigvalsh(XXT_onehot))[::-1]
eig_emb = np.sort(np.linalg.eigvalsh(XXT_emb))[::-1]

# ----- Cumulative energy -----
cum_onehot = np.cumsum(eig_onehot / eig_onehot.sum())
cum_emb = np.cumsum(eig_emb / eig_emb.sum())

# ----- Plot -----
plt.figure()
plt.plot(cum_onehot, label="One-hot (2V)")
plt.plot(cum_emb, label="Embedding (2d)")
plt.xlabel("k (number of top eigenvalues)")
plt.ylabel("Cumulative explained energy")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.title("Cumulative energy: one-hot vs embedding")
plt.show()
