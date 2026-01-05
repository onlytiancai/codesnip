---
# Slidev deck meta
# Learn more: https://sli.dev/guide/syntax.html
title: 最小二乘法：线性代数、几何与代码
---

# 最小二乘法

<div class="pt-8"></div>

背景：线性代数原理、几何直觉与 Python 实现贯穿全篇


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


---

# 问题定义与目标

- 设计矩阵 $X \in \mathbb{R}^{n \times p}$，响应向量 $y \in \mathbb{R}^n$
- 模型：$y = X\beta + \varepsilon$（$\varepsilon$ 为噪声）
- 目标：最小化平方损失 $L(\beta) = \|y - X\beta\|_2^2$

> 建立统一符号与问题背景，为后续线代与几何讨论打下基础

---

# 线性代数表述与正规方程

- 正规方程：$X^T X \\beta = X^T y$
- 列满秩 $\\mathrm{rank}(X) = p$ 时 $X^T X$ 可逆
- 闭式解：$\\hat{\\beta} = (X^T X)^{-1} X^T y$（满秩情形）

> 给出核心线代框架与解的存在性条件

---

# 几何直觉：投影与残差正交

- 列空间 $\\mathrm{span}(X)$：可表示的向量集合
- 投影：$X\\hat{\\beta}$是 $y$ 在 $\\mathrm{span}(X)$ 上的正交投影
- 残差 $r = y - X\\hat{\\beta}$与 $\\mathrm{span}(X)$ 正交 $X^T r = 0$

> 以投影与正交性加深理解最小化的几何本质

---

# 正规方程推导与凸性

- 梯度：$\\nabla_{\\beta} L(\\beta) = -2X^T(y - X\\beta) = 0$ ⇒ $X^T X \\beta = X^T y$
- $L(\\beta)$ 为凸函数；解唯一性与 $\\mathrm{rank}(X)$、$\\mathrm{cond}(X^T X)$ 有关

> 将直觉与公式对应，强调凸性与唯一性

---

# QR 分解求解与稳定性

- $X = QR$（$Q$ 正交，$R$ 上三角），解 $R\\beta = Q^T y$
- 避免直接形成 $(X^T X)$，数值更稳定
- Householder vs. Gram-Schmidt 的差异与稳定性

> 工程上更稳健的求解路径

---

# SVD 与伪逆：统一视角

- $X = U\\Sigma V^T$，伪逆 $X^+ = V\\Sigma^+ U^T$
- $\\hat{\\beta} = X^+ y$，处理秩亏与病态
- 奇异值刻画尺度与可辨识性

> 通过 SVD 统一理解与稳健解法

---

# 数值稳定性与病态性

- 条件数 $\\kappa(X)$ 或 $\\kappa(X^T X)$
- 归一化/标准化改善数值表现
- 特征缩放与正则化配合

> 数值稳定性的诊断与缓解手段

---

# 正则化：岭回归（Tikhonov）

- $\\hat{\\beta} = (X^T X + \\lambda I)^{-1} X^T y$
- 缓解多重共线性，提高泛化；$\\lambda$ 控制偏差-方差
- 交叉验证选 $\\lambda$；配合标准化

> 在 LS 框架内自然过渡到正则化

---

# 加权与约束的变体

- 加权最小二乘：$\\min \\|W^{1/2}(y - X\\beta)\\|^2$
- 一般式：$\\|y - X\\beta\\|^2 + \\lambda \\|\\Gamma\\beta\\|^2$
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

