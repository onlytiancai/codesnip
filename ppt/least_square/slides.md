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

  当矩阵 $X$ 列满秩 $\\mathrm{rank}(X) = p$ 时，矩阵 $X^T X$ 是正定且可逆的。这是得到唯一解的前提。
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

- 梯度：$\nabla_{\beta} L(\beta) = -2X^T(y - X\beta) = 0$ ⇒ $X^T X \beta = X^T y$
- $L(\beta)$ 为凸函数；解唯一性与 $\mathrm{rank}(X)$、$\mathrm{cond}(X^T X)$ 有关

> 将直觉与公式对应，强调凸性与唯一性

---

# QR 分解求解与稳定性

- $X = QR$（$Q$ 正交，$R$ 上三角），解 $R\beta = Q^T y$
- 避免直接形成 $(X^T X)$，数值更稳定
- Householder vs. Gram-Schmidt 的差异与稳定性

> 工程上更稳健的求解路径

---

# SVD 与伪逆：统一视角

- $X = U\Sigma V^T$，伪逆 $X^+ = V\Sigma^+ U^T$
- $\hat{\beta} = X^+ y$，处理秩亏与病态
- 奇异值刻画尺度与可辨识性

> 通过 SVD 统一理解与稳健解法

---

# 数值稳定性与病态性

- 条件数 $\kappa(X)$ 或 $\kappa(X^T X)$
- 归一化/标准化改善数值表现
- 特征缩放与正则化配合

> 数值稳定性的诊断与缓解手段

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

