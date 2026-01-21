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
Y_indices = []  # 存储真实标签的索引

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

# ---------------------------
# 5. Softmax + 交叉熵 方法
# ---------------------------
print("=" * 60)
print("Softmax + Cross Entropy Loss 方法")
print("=" * 60)

# 初始化参数
np.random.seed(42)
W_softmax = np.random.randn(2*V, V) * 0.01  # (2V, V)
b_softmax = np.zeros(V)                      # (V,)

# 超参数
learning_rate = 0.01
num_epochs = 100

def softmax(z):
    """数值稳定的softmax"""
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """交叉熵损失"""
    m = y_true.shape[0]
    # y_pred: (m, V), y_true: (m, V)
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss

# 训练
print(f"\nTraining with {X.shape[0]} samples...")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")
print(f"W_softmax shape: {W_softmax.shape}, b_softmax shape: {b_softmax.shape}\n")

losses = []

for epoch in range(num_epochs):
    # 前向传播
    logits = X @ W_softmax + b_softmax  # (T, V)
    probs = softmax(logits)              # (T, V)
    
    # 计算损失
    loss = cross_entropy_loss(Y, probs)
    losses.append(loss)
    
    # 反向传播
    # dL/dlogits = probs - Y  (对于交叉熵)
    dlogits = probs - Y  # (T, V)
    
    # dL/dW = X^T @ dlogits
    dW = X.T @ dlogits  # (2V, V)
    
    # dL/db = sum(dlogits)
    db = np.sum(dlogits, axis=0)  # (V,)
    
    # 更新参数
    W_softmax -= learning_rate * dW
    b_softmax -= learning_rate * db
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs}, Loss: {loss:.6f}")

print(f"\nFinal Loss: {losses[-1]:.6f}")

# ---------------------------
# 6. 预测函数 (Softmax版)
# ---------------------------
def predict_softmax(w1, w2):
    """使用softmax模型预测"""
    x = np.concatenate([one_hot(w1), one_hot(w2)])
    logits = x @ W_softmax + b_softmax
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return id2word[idx], probs[idx]

# ---------------------------
# 7. 演示预测
# ---------------------------
print("\n" + "=" * 60)
print("Softmax 模型预测结果")
print("=" * 60)

tests = [
    ("我", "喜欢"),
    ("喜欢", "学习"),
    ("学习", "是"),
    ("是", "一个"),
    ("一个","方法"),
    ("方法","可以"),    
]

for w1, w2 in tests:
    pred, conf = predict_softmax(w1, w2)
    print(f"{w1} {w2} -> {pred:6s} (confidence: {conf:.4f})")

# ---------------------------
# 8. 评估：训练集准确率
# ---------------------------
print("\n" + "=" * 60)
print("模型评估 - 训练集准确率")
print("=" * 60)

logits = X @ W_softmax + b_softmax
probs = softmax(logits)
predictions = np.argmax(probs, axis=1)
true_labels = np.argmax(Y, axis=1)
accuracy = np.mean(predictions == true_labels)
print(f"Training Accuracy: {accuracy:.4f} ({np.sum(predictions == true_labels)}/{len(true_labels)})")
