---
title: 最小二乘法：线性代数、几何与代码
theme: seriph
background: https://cover.sli.dev
---

# 最小二乘法

线性代数原理、几何直觉与 Python 实现贯穿全篇

---

# 目录

<div grid="~ cols-2 gap-4">
  <div>

  - 1、问题定义与目标
  - 2、线性代数表述与正规方程
  - 3、几何直觉（投影与残差）
  - 4、数值方法（QR / SVD）

  </div>

  <div>

  - 5、正则化与加权
  - 6、代码实现（NumPy / Scikit-Learn）
  - 7、评估与诊断
  - 8、常见问题与陷阱

  </div>
</div>

<style>

li {
  @apply m-2 p-6 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 list-none;
}
</style>

<!--
本节课的结构分为四个部分：首先定义问题，然后深入数学原理，接着探讨数值求解方法，最后是实战代码和模型诊断。我们将覆盖从理论推导到工程应用的完整流程。
-->
---

# 问题定义与目标

在开始深入算法之前，让我们首先明确我们面临的数据结构、线性假设以及我们的核心优化目标。

<div grid="~ cols-3 gap-4">
  <div>

  ### 📊 观察数据

  我们拥有一个设计矩阵 $X \in \mathbb{R}^{n \times p}$，其中包含 $n$ 个样本和 $p$ 个特征。响应变量被表示为向量 $y \in \mathbb{R}^n$ 。这是所有监督学习问题的起点。

  </div>
  <div>

  ### ⚙️ 模型设定

  我们假设输入与输出之间存在线性关系，即 $y = X\beta + \varepsilon$。这里 $\beta$ 是我们需要求解的系数向量，而 $\varepsilon$ 代表了不可观测的随机噪声或测量误差。
  </div>
  <div>

  ### 🎯 目标函数

  我们的目标是找到系数 $\beta$，使得预测值与真实值之间的差异最小化。形式上，我们最小化平方损失函数：$L(\beta) = \|y - X\beta\|^2$，即残差平方和。
  </div>
</div>

<!--
首先，我们定义问题空间。X是我们的特征数据，y是目标。我们假设线性关系，并试图通过最小化预测误差的平方和来找到最佳的系数Beta。这不仅仅是公式，更是我们对世界的线性假设。
-->
---

# 线性代数表述与正规方程

明确目标后，我们可以利用矩阵微积分将其转化为一个优雅的线性方程组求解问题。

<div class="m-2 p-6 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

## 正规方程

$X^T X \beta = X^T y$

</div>

<div grid="~ cols-2 gap-4">
  <div class="m-2 p-6 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">

  ### ✅ 可解条件

  当矩阵 $X$ 列满秩 $\mathrm{rank}(X) = p$ 时，矩阵 $X^T X$ 是正定且可逆的。这是得到唯一解的前提。
  </div>
  <div class="m-2 p-6 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">

  ### 🔑 闭式解
  在满足满秩条件下，我们可以直接写出解析解

  $\hat{\beta} = (X^T X)^{-1} X^T y$
  </div>
</div>

<!--
正规方程是最小二乘法的核心代数形式。它的核心思想是将求导置零转化为矩阵乘法。如果特征矩阵是满秩的，我们甚至可以直接写出它的逆，得到完美的闭式解。
-->
---

# 几何直觉：投影与残差正交

代数公式背后隐藏着优美的几何结构：最小二乘法本质上是一个正交投影过程。

<div class="grid grid-cols-[1fr_300px]">
  <div class="p-4">

### 1. 列空间 (Column Space)

所有可能的预测值 $X\beta$ 构成了 $X$ 列向量张成的子空间 $\mathrm{span}(X)$。真实值 $y$ 通常不在此空间内。

### 2. 投影观点 (Projection)

最佳估计值 $X\hat{\beta}$ 实际上就是向量 $y$ 在 $\mathrm{span}(X)$ 子空间上的正交投影，距离最近。

### 3. 残差正交 (Orthogonality)
残差向量 $r = y - X\hat{\beta}$ 垂直于该平面，即 $X^T r = 0$，误差无法再被 $X$ 解释。

  </div>
  <div>

<svg width="400" height="300" viewBox="0 0 600 400" >
  
  <!-- span(X) 平面 -->
  <polygon points="150,120 450,100 420,300 120,320" 
           fill="rgba(100, 150, 200, 0.3)" 
           stroke="#4a90e2" 
           stroke-width="3"
           opacity="0.8"/>
  
  <!-- 从原点到平面上的投影点 Xβ -->
  <line x1="150" y1="250" x2="280" y2="180" 
        stroke="#9d3ce7ff" 
        stroke-width="4" 
        marker-end="url(#arrowhead)"/>
  <text x="290" y="175" font-size="4" font-weight="bold" fill="#9d3ce7ff">Xβ</text>
  
  <!-- 从原点到真实数据点 y -->
  <line x1="150" y1="250" x2="250" y2="10" 
        stroke="#27ae60" 
        stroke-width="4" 
        marker-end="url(#arrowhead)"/>
  <text x="220" y="100" font-size="4" font-weight="bold" fill="#27ae60">y</text>
  
  <!-- 残差向量 r (垂直于平面) -->
  <line x1="280" y1="180" x2="250" y2="10" 
        stroke="#f39c12" 
        stroke-width="3" 
        stroke-dasharray="5,5"
        marker-end="url(#arrowhead)"/>
  <text x="270" y="100" font-size="4" font-weight="bold" fill="#f39c12">r</text>
   
  <!-- 平面标签 -->
  <text x="350" y="280" font-size="4" fill="#4a90e2" font-style="italic">span(X)</text>
  
  <!-- 原点 -->
  <circle cx="150" cy="250" r="4" fill="#000000"/>
  <text x="155" y="270" font-size="6" fill="#000000">0</text>
  
  <!-- 箭头标记定义 -->
  <defs>
    <marker id="arrowhead" markerWidth="6" markerHeight="4" 
            refX="5" refY="2" orient="auto">
      <polygon points="0 0, 6 2, 0 4" fill="currentColor"/>
    </marker>
  </defs>
  
</svg>

  </div>
</div>

<!--
几何上，最小二乘法就是寻找一个正交投影。我们的真实数据 y 往往飘在特征空间之外，我们在平面上寻找一个影 Xβ，使得它离 y 最近。此时，连接它们的残差线垂直于整个平面。
-->

---

# 正规方程推导与凸性

通过简单的微积分推导，我们可以确信局部极值即为全局最优解。


<div grid="~ cols-2 gap-4">
  <div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

### 1. 梯度设为零

$\nabla_{\beta} L(\beta) = -2X^T(y - X\beta) = 0$ 

  </div>
  <div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

### 2. 整理得到

$X^T X \beta = X^T y$

  </div>
</div>

<div class="bg-gray-100 m-2 p-4 dark:bg-gray-800 text-gray-700 dark:text-gray-300">

### 🔍 凸性分析 (Convexity)

损失函数 $L(\beta)$ 的二阶导数（Hessian 矩阵）为 $2X^T X$。
这是一个半正定矩阵 (PSD)，意味着 $L(\beta)$ 是一个凸碗状函数，保证了任何临界点都是全局最小值。解的唯一性仅取决于 X 是否满秩。

解唯一性与 $\mathrm{rank}(X)$、$\mathrm{cond}(X^T X)$ 有关

</div>

<!--
推导过程其实很简单：对Beta求导并令其为0。关键点在于二阶导数 X转置X 是半正定的，这保证了我们的损失函数是凸的，没有局部陷阱，只要梯度为0就是全球最低点。
-->

---

# QR 分解求解与稳定性

在实际工程计算中，为了数值稳定性，我们往往绕过正规方程，转而使用矩阵分解技术。

<div grid="~ cols-2 gap-4">
  <div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

### QR 分解

$X = QR$

<span class="text-gray-400 text-sm">$Q$: 正交矩阵, $R$: 上三角矩阵</span>

  </div>
  <div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

### 求解方程

$R\beta = Q^T y$

<span class="text-gray-400 text-sm">利用回代法 (Back-substitution) 快速求解</span>

  </div>
</div>


### 🌟 为什么使用 QR？

- **避免数值爆炸**：<span class="text-gray-400">直接计算 $X^T X$ 会导致条件数平方 $κ^2$，极大地放大浮点误差。QR 方法保持了 $κ(X)$ 的量级。</span>
- **实务建议**：<span class="text-gray-400">Householder 变换通常比 Gram-Schmidt 更稳定，是现代线性代数库（如 LAPACK）的首选实现。</span>

<!--
直接计算X转置X虽然公式漂亮，但在计算机里是个坏主意，因为它会平方条件数，导致精度丢失。QR分解将矩阵拆分为正交和上三角部分，让我们能更稳定地求解，是工业界的标准做法。
-->
---

# SVD 与伪逆：统一视角

当矩阵秩亏或接近奇异时，奇异值分解 (SVD) 提供了最强大的通用求解工具。

<div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

$X = U\Sigma V^T$
</div>

<div grid="~ cols-3 gap-4">
<div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

**$U$ (左奇异向量)**

<span class="text-gray-400">正交矩阵，数据空间基</span>

</div>
<div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

**$\Sigma$ (奇异值)**

<span class="text-gray-400">对角矩阵，刻画重要性尺度</span>

</div>
<div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

**$V$ (右奇异向量)**

<span class="text-gray-400">正交矩阵，特征空间基</span>

</div>
</div>

<div class="text-center">

通用接伪逆 $X^+ = V\Sigma^+ U^T$，$\hat{\beta} = X^+ y$

<span class="text-gray-400">能够自动处理 X 不满秩的情况（通过将接近零的奇异值倒数设为零）。</span>

</div>

<!--
SVD 是矩阵分解的瑞士军刀。无论矩阵多糟糕（秩亏、病态），SVD 都能通过伪逆给出最小范数解。它通过奇异值 Σ 告诉我们哪些方向的数据是重要的，哪些是噪声。
-->

---

# 数值稳定性与病态性

在工程实践中，判断一个线性系统是否“健康”至关重要，这直接决定了结果的可信度。

<div grid="~ cols-2 gap-4">
  <div>
  <div class="m-2 p-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">

 1. 诊断病态

<span class="text-gray-400 text-sm">如果 κ(X) 过大，矩阵被称为“病态”的，求逆计算将极不稳定。</span>
  </div>
  <div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">

2. 数据标准化 (Normalization)

<span class="text-gray-400 text-sm">对特征进行归一化（如 Z-score 缩放），使其具有相似的尺度，能显著降低条件数。</span>
  </div>
  <div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">

3. 配合正则化

<span class="text-gray-400 text-sm">即使数据已缩放，若特征间存在强相关性，仍需引入正则化来改善数值稳定性。</span>
  </div>
  </div>
  <div class="m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

### 条件数 (Condition Number)

$\kappa(X)$ 或 $\kappa(X^T X)$

<span class="text-gray-400 text-sm text-left">衡量输入微小扰动对输出的影响放大倍数。$\kappa(X) > 1000$ 警惕！ </span>

  </div>
</div>

- 条件数 或 $
- 归一化/标准化改善数值表现
- 特征缩放与正则化配合

<!--
条件数是我们的警报器。当它很大时，意味着矩阵接近奇异，计算结果不可信。最简单的解药是数据预处理：把所有特征缩放到同一尺度。如果这还不够，就得请出正则化了。
-->

---

# 正则化：岭回归（Tikhonov）

- $\hat{\beta} = (X^T X + \lambda I)^{-1} X^T y$
- 缓解多重共线性，提高泛化；$\lambda$ 控制偏差-方差
- 交叉验证选 $\lambda$；配合标准化

> 在 LS 框架内自然过渡到正则化

---

# 加权与约束的变体

- 加权最小二乘：$\min \|W^{1/2}(y - X\beta)\|^2$
- 一般式：$\|y - X\beta\|^2 + \lambda \|\Gamma\beta\|^2$
- 简介 LASSO（L1）与稀疏偏好

> 展示 LS 生态的常见扩展

---

# 代码实现：NumPy lstsq 示例

```py
import numpy as np

# 合成数据
data_rng = np.random.default_rng(0)
n, p = 100, 2
X = np.c_[np.ones(n), np.linspace(0, 1, n)]
true_beta = np.array([1.0, 2.0])
y = X @ true_beta + 0.1 * data_rng.normal(size=n)

# 最小二乘（NumPy）
beta_hat, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
print(beta_hat, residuals, rank, s)
```

- 返回值含奇异值与秩信息，便于诊断
- 可与闭式解/QR 解对比

---

# 代码实现：Scikit-Learn 示例

```py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression(fit_intercept=True)
model.fit(X[:, 1].reshape(-1, 1), y)
print(model.coef_, model.intercept_)

y_pred = model.predict(X[:, 1].reshape(-1, 1))
print('MSE=', mean_squared_error(y, y_pred))
print('R2=', r2_score(y, y_pred))
```

- 常用流程与评估：$R^2$ / $MSE$
- 可与标准化、管道、交叉验证结合

---

# 评估与诊断：残差与杠杆

- 残差分析：均值接近 0，观察是否有结构性模式
- Hat 矩阵 $H = X(X^T X)^{-1}X^T$；杠杆点与影响度
- 评估指标：$MSE$、$MAE$、$R^2$；训练/验证划分

> 将理论与实践闭环到质量控制

---
layout: cover
class: text-center
---

# 谢谢观看

回顾：线代框架、几何直觉、QR/SVD、实践代码、诊断与正则化

