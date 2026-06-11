import numpy as np
import sys

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

def one_hot(word):
    v = np.zeros(V)
    v[word2id[word]] = 1.0
    return v

# ---------------------------
# 3. 构造 2-gram
# ---------------------------
X_list, Y_list = [], []

for sent in sentences:
    words = sent.split()
    for i in range(len(words) - 2):
        x = np.concatenate([one_hot(words[i]), one_hot(words[i + 1])])
        y = one_hot(words[i + 2])
        X_list.append(x)
        Y_list.append(y)

X = np.stack(X_list, axis=0)   # (T, 2V) - T个样本，每个样本2V维特征
Y = np.stack(Y_list, axis=0)   # (T, V) - T个样本，每个样本V维标签

print(f"X shape: {X.shape}  (样本数={X.shape[0]}, 特征维={X.shape[1]})")
print(f"Y shape: {Y.shape}  (样本数={Y.shape[0]}, 标签维={Y.shape[1]})")
print(f"词表大小 V={V}, 特征维度 2V={2*V}")

# ---------------------------
# 4. 稳定最小二乘（使用公式实现）
# ---------------------------

# 特征缩放（极关键）
X_scale = np.linalg.norm(X, axis=0, keepdims=True)
X_scale[X_scale == 0] = 1.0  # 防止除零
Xn = X / X_scale

# 使用公式 W^T = Y^T X (X^T X + λI)^(-1) 或等价形式
# 当特征维度(2V)大于样本数(T)时，使用对偶形式更稳定
lam = 1e-2  # 正则化参数

# 计算 X^T X 和 Y^T X
XTX = np.dot(Xn.T, Xn)  # (2V, 2V)
YTX = np.dot(Y.T, Xn)   # (V, 2V)

# 添加正则化项 (X^T X + λI)
XTX_reg = XTX + lam * np.eye(Xn.shape[1])

# 计算 (X^T X + λI)^(-1)
XTX_inv = np.linalg.inv(XTX_reg)

# 使用公式计算 W^T: W^T = Y^T X (X^T X)^(-1)
# W = [(X^T X)^(-1)]^T X^T Y = (X^T X)^(-1) X^T Y (因为对称矩阵)
WT = np.dot(YTX, XTX_inv)  # (V, 2V)

# W 是 WT 的转置，但在预测时使用 WT (形状: V x 2V)
W = WT.T  # (2V, V)



# 检查 W 矩阵的数值稳定性
if np.any(np.isnan(W)) or np.any(np.isinf(W)):
    print("警告：W 矩阵包含 NaN 或 Inf")
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

# ---------------------------
# 5. 预测
# ---------------------------
def predict(w1, w2):
    x = np.concatenate([one_hot(w1), one_hot(w2)])
    
    # 安全的特征缩放：避免除零警告
    x_scaled = np.zeros_like(x)
    mask = X_scale.flatten() > 1e-10  # 只缩放训练时出现过的特征
    x_scaled[mask] = x[mask] / X_scale.flatten()[mask]
    
    # 检查是否遇到未训练的组合
    if np.all(x_scaled == 0):
        # 使用随机预测作为回退
        random_idx = np.random.randint(0, V)
        return id2word[random_idx]
    
    # y_hat = x_scaled @ W^T 得到 (23,) 的预测向量
    # 或者 y_hat = W^T @ x_scaled (转置后相乘)
    y_hat = np.dot(W.T, x_scaled)
    return id2word[int(np.argmax(y_hat))]


def generate_sentence(start_word, num_words):
    """
    生成以start_word开头，长度为num_words的句子
    """
    if num_words < 1:
        return ""
    
    sentence = [start_word]
    
    # 如果只需要一个词，直接返回
    if num_words == 1:
        return start_word
    
    # 需要预测下一个词，但模型需要两个前序词
    # 如果第一个词不在词表中，随机选择一个词
    if start_word not in word2id:
        print(f"警告：起始词'{start_word}'不在词表中，将随机选择")
        start_word = vocab[np.random.randint(0, V)]
        sentence = [start_word]
    
    # 生成第二个词（随机选择，因为模型需要两个词来预测）
    second_word = vocab[np.random.randint(0, V)]
    sentence.append(second_word)
    
    # 逐个预测后续的词
    while len(sentence) < num_words:
        w1 = sentence[-2]
        w2 = sentence[-1]
        next_word = predict(w1, w2)
        sentence.append(next_word)
    
    return " ".join(sentence)


# 命令行参数解析
if __name__ == "__main__":
    if len(sys.argv) != 3:
        tests = [
            ("我", "喜欢"),
            ("机器", "学习"),
            ("这个", "模型"),
            ("我们", "用"),
            ("我", "是"),
            ("是", "一个"),
            ("一个", "模型"),
            ("模型", "可以"),
        ]

        for w1, w2 in tests:
            print(f"{w1} {w2} -> {predict(w1, w2)}")
            
        print("用法: python 002_ls_text.py <起始单词> <句子单词数>")
        print("示例: python 002_ls_text.py 我 5")
        print("\n可用词表:", " ".join(vocab))        
        sys.exit(1)
    
    start_word = sys.argv[1]
    try:
        num_words = int(sys.argv[2])
    except ValueError:
        print("错误：句子单词数必须是整数")
        sys.exit(1)
    
    if num_words < 1:
        print("错误：句子单词数必须大于0")
        sys.exit(1)
    
    # 生成句子
    sentence = generate_sentence(start_word, num_words)
    print(f"生成的句子: {sentence}")
