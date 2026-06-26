"""
演示 Transformer 注意力机制 - 例子1：代词指代

句子：The animal didn't cross the street because it was tired.
核心：当模型处理 "it" 时，Q(it) · K(animal) 应该远大于 Q(it) · K(street)

embedding / W_Q / W_K / W_V 全部写死为 4 维向量。
"""

import math

# ----------------------------------------------------------------------
# 1. Token 列表
# ----------------------------------------------------------------------
tokens = ["The", "animal", "didn't", "cross", "the", "street", "it"]

# ----------------------------------------------------------------------
# 2. 4 维 embedding
#    dim 0: 是不是名词
#    dim 1: 是不是有生命
#    dim 2: 能不能感到疲劳
#    dim 3: 是不是代词
# ----------------------------------------------------------------------
embeddings = {
    "The":    [0.0, 0.0, 0.0, 0.0],
    "animal": [1.0, 1.0, 1.0, 0.0],   # 名词 + 有生命 + 能累
    "didn't": [0.0, 0.0, 0.0, 0.0],
    "cross":  [0.0, 0.0, 0.0, 0.0],
    "the":    [0.0, 0.0, 0.0, 0.0],
    "street": [1.0, 0.0, 0.0, 0.0],   # 名词 + 无生命 + 不能累
    "it":     [0.0, 0.0, 0.0, 1.0],   # 代词
}

# ----------------------------------------------------------------------
# 3. W_Q / W_K / W_V (4x4)
#
#    目标：
#      Q(it)     = [0, 1, 1, 0]   "想找有生命 且 能累的东西"
#      K(animal) = [1, 1, 1, 0]   名词 + 有生命 + 能累
#      K(street) = [1, 0, 0, 0]   名词 + 无生命 + 不能累
#
#    关键：K(animal) dim1+dim2 都为 1，K(street) dim1+dim2 都为 0
#         → Q(it)·K(animal) = 2,  Q(it)·K(street) = 0
# ----------------------------------------------------------------------
W_Q = [
    [0.0, 0.0, 0.0, 0.0],   # Q dim0
    [0.0, 0.0, 0.0, 1.0],   # Q dim1 = 代词，确保乘 it 得到 `生命` 为 1
    [0.0, 0.0, 0.0, 1.0],   # Q dim2 = 代词，确保乘 it 得到 `能累` 为 1
    [0.0, 0.0, 0.0, 0.0],   # Q dim3
]

# 关键：K(animal) 在 dim1 和 dim2 上都为 1 → 点积 2
#      K(street) 在 dim1 和 dim2 上都为 0 → 点积 0
#      Q(it) 同时 "找" dim1 和 dim2，两个特征都贡献了差距

W_K = [
    [1.0, 0.0, 0.0, 0.0],   # K dim0 = dim0(名词)，确保乘 animal 和 street 是名词
    [0.0, 1.0, 0.0, 0.0],   # K dim1 = dim1(有生命)，确保乘 animal 是有生命，乘 street 是无生命
    [0.0, 0.0, 1.0, 0.0],   # K dim2 = dim2(能累)，确保乘 animal 是能累，乘 street 是不能累
    [0.0, 0.0, 0.0, 0.0],   # K dim3
]

W_V = [
    [1.0, 0.0, 0.0, 0.0],   # V dim0 = dim0(名词身份)
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
]

def matvec(W, x): # 矩阵乘向量，矩阵的每一行和向量做点击后相乘
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]

Q = {t: matvec(W_Q, embeddings[t]) for t in tokens}
K = {t: matvec(W_K, embeddings[t]) for t in tokens}
V = {t: matvec(W_V, embeddings[t]) for t in tokens}

# ----------------------------------------------------------------------
# 4. Q(it) · K(t)
# ----------------------------------------------------------------------
def dot(a, b): # 向量乘向量，行乘列，点积
    return sum(x * y for x, y in zip(a, b))

query_token = "it"
scores = {t: dot(Q[query_token], K[t]) for t in tokens}

# ----------------------------------------------------------------------
# 5. Softmax
# ----------------------------------------------------------------------
def softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [e / s for e in exps]

attn_weights = dict(zip(tokens, softmax(list(scores.values()))))

# ----------------------------------------------------------------------
# 6. 加权求和 output("it")
# ----------------------------------------------------------------------
output = [0.0] * 4
for t in tokens:
    for i in range(4):
        output[i] += attn_weights[t] * V[t][i]

# ----------------------------------------------------------------------
# 7. 打印
# ----------------------------------------------------------------------
def fmt(v):
    return "[" + ", ".join(f"{x:+.2f}" for x in v) + "]"

print("=" * 64)
print(' 句子: "The animal didn\'t cross the street because it was tired."')
print("=" * 64)
print(f"\n embedding (4 维):")
for t in tokens:
    print(f"   {t:<10} {fmt(embeddings[t])}")

print(f"\n Q({query_token})  = W_Q · embedding(it)    = {fmt(Q[query_token])}")
print(f" K(animal)        = W_K · embedding(animal) = {fmt(K['animal'])}")
print(f" K(street)        = W_K · embedding(street) = {fmt(K['street'])}")

print(f"\n {'token':<10}{'Q·K':>8}{'softmax':>10}  权重条")
print("-" * 60)
for t in tokens:
    bar = "█" * int(attn_weights[t] * 50)
    print(f" {t:<10}{scores[t]:>8.2f}{attn_weights[t]:>10.4f}  {bar}")

print(f"\n output({query_token}) = Σ α·V = {fmt(output)}")
print(f"\n 结论:")
print(f"   Q(it)·K(animal) = {scores['animal']:.2f}")
print(f"   Q(it)·K(street) = {scores['street']:.2f}")
print(f"   α(animal) = {attn_weights['animal']:.3f}  >>  α(street) = {attn_weights['street']:.3f}")
print(" → 模型把 'it' 解读为 'animal'，而不是 'street'")
