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

<div grid="~ cols-3 gap-4">
  <div class="col-start-1 col-end-3">
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
  <div class="col-start-3 col-end-4 m-2 p-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

### 条件数 (Condition Number)

$\kappa(X)$ 或 $\kappa(X^T X)$

<span class="text-gray-400 text-sm">衡量输入微小扰动对输出的影响放大倍数。$\kappa(X) > 1000$ 警惕！ </span>

  </div>
</div>

<!--
条件数是我们的警报器。当它很大时，意味着矩阵接近奇异，计算结果不可信。最简单的解药是数据预处理：把所有特征缩放到同一尺度。如果这还不够，就得请出正则化了。
-->

---

# 正则化：岭回归（Tikhonov）

为了抑制过拟合和解决病态矩阵问题，我们引入结构性偏好 $\lambda$。

<div class="p-4 mx-20 my-4 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-center">

$\hat{\beta} = (X^T X + \lambda I)^{-1} X^T y$

</div>

<div grid="~ cols-2 gap-4">

<div>

### 核心作用

- **数学上**：通过给对角线元素加上 $\lambda$，强制矩阵满秩可逆，消除多重共线性带来的奇点。
- **统计上**：引入偏差以换取方差的大幅降低 (Bias-Variance Tradeoff)，提高模型在未知数据上的泛化能力。

</div>
  
<div>

### $\lambda$ 的选择

- **经验法则**：使用交叉验证 (Cross-Validation) 扫描对数尺度的 $\lambda$ 值（如 0.01, 0.1, 1, 10）。
- **前提**：应用岭回归前，必须先对特征进行标准化，否则 $\lambda$ 对不同尺度的特征惩罚力度不均。

</div>
</div>

<!--
岭回归通过在对角线上加一个小常数 Lambda，奇迹般地解决了不可逆问题。这不仅是数学技巧，更是机器学习核心思想——偏差方差权衡的体现。记住，用它之前一定要先做标准化。
-->

---

# 加权与约束的变体

基础最小二乘法可以灵活扩展，以适应更复杂的现实场景。

<div grid="~ cols-3 gap-4">
  <div>

### ⚖️ 加权最小二乘 (WLS)

$$\min \|W^{1/2}(y - X\beta)\|^2$$

<span class="text-gray-400 text-sm">当某些样本比其他样本更重要，或测量误差方差不一致（异方差性）时，给每个误差项赋予不同权重。</span>

  </div>
  <div>

### ⛓️ 广义 Tikhonov


$$\min \|y - X\beta\|^2 + \lambda \|\Gamma\beta\|^2$$

<span class="text-gray-400 text-sm">不仅惩罚系数大小，还可以惩罚系数之间的差异（如平滑约束），通过设计矩阵 $\Gamma$ 实现。</span>

  </div>
  <div>

### 💎 稀疏偏好 (LASSO)

$$ \min \|y - X\beta\|^2 + \lambda \| \beta \|_1 $$

<span class="text-gray-400 text-sm">使用 L1 范数正则化，能强制使不重要的特征系数变为 0，从而实现自动特征选择。</span>

  </div>
</div>

<!--
LS家族很庞大。如果数据质量不一，用加权LS；如果想要系数平滑，用广义Tikhonov；如果想做特征筛选，LASSO是首选。根据场景灵活变通是关键。
-->
---

# 代码实现：NumPy lstsq 示例

NumPy 提供了基于 SVD 的低级接口，适合深入理解解的细节。

<div class="grid grid-cols-3 gap-4">
  <div class="col-start-1 col-end-3">

```py
import numpy as np
# 1. 构造合成数据 (X, y)
X = np.c_[np.ones(10), np.random.rand(10)]
y = X @ np.array([2, 5]) + np.random.normal(size=10)
# 2. 求解 Least Squares
beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
# 3. 输出结果
print(f"Coefficients: {beta}")
```

  </div>
  <div class="col-start-3 col-end-4">

  ### 💡 关键点

  **返回值信息丰富：**

  <div class="text-gray-400 text-sm">

  除了系数 beta，它还返回：
  - residuals: 残差平方和
  - rank: 矩阵 X 的秩
  - s: X 的奇异值，用于诊断病态性
  </div>

  </div>
</div>

<!--
NumPy的lstsq函数是底层的强力工具。它不仅给你答案，还告诉你问题的性质（比如秩和奇异值）。注意构造数据时，我们通常需要手动添加一列全1作为截距项。
-->

---

# 代码实现：Scikit-Learn 示例

在机器学习工作流中，Sklearn 提供了更加封装和易用的接口。

<div class="grid grid-cols-3 gap-4">
<div class="col-start-1 col-end-3">

```py
from sklearn.linear_model import LinearRegression
# 1. 初始化模型 (自动处理截距)
model = LinearRegression(fit_intercept=True)
# 2. 拟合数据
model.fit(X, y)
# 3. 获取参数与评估
print(f"Coef: {model.coef_}, Bias: {model.intercept_}")
print(f"R^2 Score: {model.score(X, y)}")
```
</div>
<div class="col-start-3 col-end-4">

### 🚀 工程优势

Pipeline 集成：

<span class="text-gray-400 text-sm">可以轻松与 StandardScaler 和 PolynomialFeatures 串联，构建完整的机器学习流水线。</span>

标准化接口：

<span class="text-gray-400 text-sm">统一的 fit/predict/score API，方便切换到 Lasso 或 Ridge 回归。</span>

</div>
</div>

<!--
Sklearn是生产环境的首选。它自动处理截距项，并且其统一的API设计让你在尝试不同模型（如Ridge或Lasso）时几乎不需要改代码。别忘了用R方分数来快速评估模型好坏。
-->
---

# 评估与诊断：残差与杠杆

模型拟合只是第一步，深入的诊断能帮助我们发现数据中的异常和模型的不足。

<div grid="~ cols-3 gap-4">
<div>

### 📉 残差分析

检查残差图 (Residual Plot)。理想的残差应是围绕 0 随机分布的白噪声。

若出现“喇叭口”形状（异方差）或曲线模式，说明模型假设可能失效。


  </div>
  <div>

### 🏗️ 杠杆与影响点

通过帽子矩阵 (Hat Matrix) H 识别杠杆点 (High Leverage)。

远离中心的点可能对模型产生不成比例的巨大影响 (Cook's Distance)，需要特别关注。

$$H = X(X^T X)^{-1}X^T$$

  </div>
  <div>

### 📊 评估指标

**$R^2$ (决定系数)**：解释了多大比例的方差。

**MSE (均方误差)**：预测误差的平方期望，对大误差敏感。

  </div>
</div>


> 将理论与实践闭环到质量控制

---
layout: cover
class: text-center
---

# 谢谢观看

回顾：线代框架、几何直觉、QR/SVD、实践代码、诊断与正则化

