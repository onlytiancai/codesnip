import numpy as np
import matplotlib.pyplot as plt

# Reconstruct variables from the given code
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

X_list = []
for sent in sentences:
    words = sent.split()
    for i in range(len(words) - 2):
        w1, w2 = words[i], words[i+1]
        x = np.concatenate([one_hot(w1), one_hot(w2)])
        X_list.append(x)

X = np.stack(X_list, axis=1)   # (2V, T)

XXT = X @ X.T

# 特征值谱
# Eigenvalues
eigvals = np.linalg.eigvalsh(XXT)
eigvals_sorted = np.sort(eigvals)[::-1]

plt.figure()
plt.semilogy(eigvals_sorted)
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue (log scale)")
plt.title("Eigenvalue spectrum of X X^T")
plt.show()

# 累计能量曲线
# ----- Eigenvalues -----
eigvals = np.linalg.eigvalsh(XXT)
eigvals = np.sort(eigvals)[::-1]   # descending

# ----- Cumulative energy -----
energy = eigvals / eigvals.sum()
cum_energy = np.cumsum(energy)

# ----- Plot -----
plt.figure()
plt.plot(cum_energy, marker='o')
plt.xlabel("k (number of top eigenvalues)")
plt.ylabel("Cumulative explained energy")
plt.title("Cumulative energy of eigenvalues of X X^T")
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()