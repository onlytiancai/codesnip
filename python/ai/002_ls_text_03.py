import numpy as np
import matplotlib.pyplot as plt

# ----- Reconstruct vocab and data -----
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

def one_hot(word):
    v = np.zeros(V)
    v[word2id[word]] = 1.0
    return v

# ----- Build X (concatenation) -----
X_concat = []
X_cbow = []

for sent in sentences:
    words = sent.split()
    for i in range(len(words) - 2):
        w1, w2 = words[i], words[i+1]
        X_concat.append(np.concatenate([one_hot(w1), one_hot(w2)]))
        X_cbow.append(one_hot(w1) + one_hot(w2))

X_concat = np.stack(X_concat, axis=1)  # (2V, T)
X_cbow = np.stack(X_cbow, axis=1)      # (V, T)

# ----- XX^T -----
XXT_concat = X_concat @ X_concat.T
XXT_cbow = X_cbow @ X_cbow.T

# ----- Eigenvalues -----
eig_concat = np.sort(np.linalg.eigvalsh(XXT_concat))[::-1]
eig_cbow = np.sort(np.linalg.eigvalsh(XXT_cbow))[::-1]

# ----- Cumulative energy -----
cum_concat = np.cumsum(eig_concat / eig_concat.sum())
cum_cbow = np.cumsum(eig_cbow / eig_cbow.sum())

# ----- Plot comparison -----
plt.figure()
plt.plot(cum_concat, label="Concatenation (2V)")
plt.plot(cum_cbow, label="CBOW sum (V)")
plt.xlabel("k (number of top eigenvalues)")
plt.ylabel("Cumulative explained energy")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.title("Cumulative energy: concatenation vs CBOW")
plt.show()
