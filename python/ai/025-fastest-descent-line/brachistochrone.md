# 最速降线与摆线：从变分法到几何之美

> 写给数学基础薄弱但有好奇心的读者。每一步都尽量拆细，关键思想用"为什么"而不是"是什么"来引出。

---

## 0. 这篇文档在讲什么

**一句话**：把一个小球从 A 点松手，让它沿一条曲线滑到 B 点，所用时间最短的曲线是什么？答案是一条叫**摆线**的曲线。这条曲线本身有一段很美的几何故事——它是一个轮子在地面上滚动时，轮缘上某一点画出的轨迹。

我们这篇文章要做三件事：

1. 解释"为什么直线不是最快的"，建立物理直觉。
2. 从零开始推导出**摆线方程**（这要用到变分法和欧拉-拉格朗日方程）。
3. 计算摆线**一拱**（参数从 0 到 2π）的**弧长**和**所围面积**。

全文需要的先修知识只有：导数、积分、三角恒等式。变分法部分我会从头讲。

---

## 1. 一个古老的问题

### 1.1 历史

1696 年，瑞士数学家**约翰·伯努利**（Johann Bernoulli）在《教师学报》上向全欧洲的数学家发出挑战：

> "我，约翰·伯努利，向全世界最杰出的数学家们致敬……我要向他们提出一个极富挑战性的问题……给定两个高度不同的点，在两点之间找一条曲线，使得质点在自身重力作用下沿该曲线从一点滑到另一点所用时间最短。"

这道题叫**最速降线问题**（Brachistochrone Problem，源自希腊语 *brakhistos*（最短）+ *khronos*（时间））。它引发了一场跨越欧陆的数学竞赛，约翰的哥哥雅各布·伯努利、牛顿、莱布尼茨、洛必达都给出了解答。答案出人意料地**不是直线**，而是一种叫**摆线（cycloid）**的曲线。

### 1.2 为什么不是直线？

你的第一反应可能是"两点之间线段最短"。但注意：题目要求**时间**最短，不是**距离**最短。

直线有两个"缺点"：

1. 在顶端时，斜率小，质点还没加速多少，已经走过了一段水平距离。
2. 在底端时，斜率大，质点虽然速度快，但只剩很短的距离可以加速。

最优策略其实是**先用一段陡坡快速加速**（哪怕为此多走了一些路径），**再在底部高速水平滑行**——这是赛车的过弯策略，也是为什么滑雪场的速降赛道是"先掉一段悬崖再铺平"。

摆线恰好满足这种"先掉得深、再水平走"的形状。它在起点处的切线是**垂直的**（这意味着质点一开始就是自由落体），在中段逐渐变平。

### 1.3 一张图先剧透

下面这张图是同一个落体问题中，三条候选路径的比较：

- 蓝色直线：从左上到右下的直线。
- 红色摆线：最优解。
- 绿色抛物线：另一个常见猜测。

可以直观看到，摆线在中段比直线"下凹得更多"，在底部又比直线更"平"。

![三条候选路径对比](figure1.png)

---

## 2. 物理准备：质点在曲线上怎么加速

要写出"时间"，我们需要两件事：每一段的速度，以及每一段的长度。

### 2.1 能量守恒（机械能守恒）

设质点从高度 $h$ 处无摩擦滑下，重力加速度 $g$。在任意时刻：

$$\frac{1}{2} m v^2 = m g h$$

其中 $h$ 是它**已经下降的高度**。所以速度大小只取决于"掉下来多少"，与走过的路径无关：

$$v = \sqrt{2 g h}$$

> 这是高中物理的能量守恒。如果忘了为什么，记住一句话：**下得越深，速度越大**。

### 2.2 弧长元素

设曲线用 $y = y(x)$ 表示，$x$ 取水平方向，$y$ **向下为正**（这是最速降线问题的标准设置：起点 $y=0$，终点 $y>0$）。

一段无限小的曲线段长度为：

$$ds = \sqrt{(dx)^2 + (dy)^2} = \sqrt{1 + \left(\frac{dy}{dx}\right)^2} \, dx$$

> 直观理解：水平走了 $dx$，竖直走了 $dy$，合成的小段长度就是 $\sqrt{dx^2 + dy^2}$。这就是著名的勾股定理在微分下的版本。

把 $y$ 简写为 $y' = dy/dx$，弧长元素记作：

$$ds = \sqrt{1 + (y')^2} \, dx$$

### 2.3 时间元素

速度 $v$ 走弧长 $ds$ 需要时间：

$$dt = \frac{ds}{v} = \frac{\sqrt{1 + (y')^2}}{\sqrt{2 g y}} \, dx$$

> 这里用了 $h = y$：当质点滑到坐标 $(x, y)$ 处时，它已经下降了 $y$（因为我们设 $y$ 向下为正，且起点在原点）。

### 2.4 总时间：我们要最小化的对象

把 $dt$ 从起点 $x=0$ 积分到终点 $x=a$，得到总时间：

$$T[y] = \int_0^a \frac{\sqrt{1 + (y')^2}}{\sqrt{2 g y}} \, dx \tag{★}$$

注意：这里的 $T$ **不是** $y$ 的普通函数——你输入一个**整条曲线** $y(x)$，它才输出一个**数**（时间）。这种"函数的函数"叫**泛函**。

我们要找的，就是让这个泛函取得最小值的曲线 $y(x)$。

---

## 3. 数学准备：变分法

### 3.1 函数 vs. 泛函

- **函数**：吃一个数，吐一个数。例如 $f(x) = x^2$。我们问"哪个 $x$ 让 $f(x)$ 最小？"——取导数令其为零。
- **泛函**：吃一个**函数**，吐一个数。例如 $T[y]$：你给一条曲线，它告诉你滑下来要多久。我们问"哪个 $y(x)$ 让 $T[y]$ 最小？"

泛函的极值问题，需要一个推广版的"令导数为零"——这就是**变分法**。

### 3.2 直观：邻域内的微小扰动

要找泛函 $J[y]$ 的极值，我们考察把最优曲线 $y(x)$ **轻微扰动**一下会怎样。

设 $y(x)$ 是最优曲线。令

$$y_\varepsilon(x) = y(x) + \varepsilon \eta(x)$$

其中 $\eta(x)$ 是任意一个**在两端点为零的平滑函数**（因为扰动不能改起点和终点），$\varepsilon$ 是一个很小的实数。

把 $y_\varepsilon$ 代入泛函：

$$J[\varepsilon] = J[y_\varepsilon]$$

如果 $y$ 是最优的，那么 $\varepsilon = 0$ 必须是 $J[\varepsilon]$ 的极值点，于是：

$$\left. \frac{d J}{d \varepsilon} \right|_{\varepsilon = 0} = 0$$

这就是"变分法的基本引理"的精神：**最优曲线在任意扰动下的一阶变分为零**。

### 3.3 推导欧拉-拉格朗日方程

设泛函形如：

$$J[y] = \int_a^b L(x, y, y') \, dx$$

其中 $L$ 是已知的"被积函数"（对最速降线，$L = \sqrt{(1+(y')^2)/(2gy)}$）。

把 $y_\varepsilon = y + \varepsilon \eta$ 代入，对 $\varepsilon$ 求导：

$$\frac{dJ}{d\varepsilon} = \int_a^b \left[ \frac{\partial L}{\partial y} \eta + \frac{\partial L}{\partial y'} \eta' \right] dx$$

> 注意：$L$ 通过 $y$ 和 $y'$ 依赖于 $\varepsilon$。$y$ 给出第一项，$y'$ 给出第二项。

对第二项做**分部积分**（$\int u dv = uv - \int v du$）：

$$\int_a^b \frac{\partial L}{\partial y'} \eta' \, dx = \left[ \frac{\partial L}{\partial y'} \eta \right]_a^b - \int_a^b \frac{d}{dx}\left( \frac{\partial L}{\partial y'} \right) \eta \, dx$$

因为 $\eta$ 在端点处为零（$y_\varepsilon$ 必须满足同样的端点条件），第一项为零：

$$\frac{dJ}{d\varepsilon} = \int_a^b \left[ \frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} \right] \eta \, dx$$

令 $\varepsilon=0$ 时此式为零，且 $\eta(x)$ 是**任意**满足端点为零的函数，那么一个深刻的结论是**括号里的东西必须处处为零**（这就是"基本引理"或"du Bois-Reymond 引理"的结论：连续函数与任意 $\eta$ 的内积为零，则函数本身为零）。

于是得到**欧拉-拉格朗日方程**：

$$\boxed{\frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} = 0} \tag{E-L}$$

这就是泛函取极值的必要条件。看起来复杂，其实就是把"令导数为零"的思路推广到了函数空间。

### 3.4 一个小例子熟悉一下

对泛函 $J[y] = \int_0^1 (y'^2) dx$，固定 $y(0)=0, y(1)=1$。

$L = (y')^2$，所以 $\partial L/\partial y = 0$，$\partial L/\partial y' = 2y'$。

E-L 方程：$0 - \frac{d}{dx}(2y') = 0$，即 $y'' = 0$。

解得 $y(x) = x$（直线）。验证：从 $(0,0)$ 到 $(1,1)$ 的最短路径就是直线——和我们"两点之间线段最短"的直觉一致。

> 但注意：这里的泛函是 $\int y'^2 dx$，**最小化的是"陡峭程度"**，不是距离也不是时间。所以"两点之间线段最短"对距离泛函 $\int \sqrt{1+(y')^2} dx$ 成立，对别的泛函就不一定——这就是最速降线问题的来源。

### 3.5 先看个数值证据

我们先用 $\int \sqrt{(1+y'^2)/(2gy)} \, dx$ 这个时间泛函，对前面提的三条候选路径数值积分，看看谁更快。结果（同一个起点 $(0,0)$ 到终点 $(\pi, 2)$，$g=1$）：

![三条路径的总滑行时间对比](figure3.png)

注意：
- **直线** $T \approx 3.63$
- **抛物线** $T \approx 11.65$（反而最慢——它在底部斜率太大，浪费了势能）
- **摆线** $T \approx 3.15$（最快，比直线快约 13.5%）

直觉告诉我们的"直线最快"是错的。下面我们用变分法严格推导出摆线为什么是最优解。

---

## 4. 应用 E-L 方程解最速降线

### 4.1 写出 $L$

从 $(★)$ 拿出被积函数（常数 $\sqrt{1/(2g)}$ 可以扔掉，因为它不影响最优曲线）：

$$L(x, y, y') = \sqrt{\frac{1 + (y')^2}{y}}$$

### 4.2 简化第一步：Beltrami 恒等式

观察到 $L$ **不显式依赖 $x$**（没有 $x$ 自己出现）。这种情况下，E-L 方程有一个**第一积分**，叫 Beltrami 恒等式：

$$L - y' \frac{\partial L}{\partial y'} = C \quad (\text{常数})$$

**怎么得到？** 把 E-L 方程两边乘以 $y'$，稍作整理。因为 $\frac{dL}{dx} = \frac{\partial L}{\partial y} y' + \frac{\partial L}{\partial y'} y''$，有：

$$0 = \left( \frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} \right) y' = \frac{dL}{dx} - \frac{d}{dx} \left( y' \frac{\partial L}{\partial y'} \right)$$

积分得 $L - y' \partial L/\partial y' = C$。**这就是 Beltrami 恒等式**。

> 它的物理直觉：$L$ 像"系统的拉格朗日量"，$y' \partial L/\partial y'$ 像"广义动量"，它们的差不显含 $x$，所以是守恒量——这就是"能量守恒"。

### 4.3 代入计算

$$L = \sqrt{\frac{1 + (y')^2}{y}}, \quad \frac{\partial L}{\partial y'} = \frac{y'}{\sqrt{y(1 + (y')^2)}}$$

于是：

$$L - y' \frac{\partial L}{\partial y'} = \frac{1 + (y')^2 - (y')^2}{\sqrt{y(1 + (y')^2)}} = \frac{1}{\sqrt{y(1 + (y')^2)}} = C$$

两边平方再取倒数：

$$y(1 + (y')^2) = \frac{1}{C^2}$$

令 $\dfrac{1}{C^2} = 2a$（$a$ 是某个正常数），得到最速降线的**常微分方程**：

$$\boxed{y(1 + (y')^2) = 2a} \tag{★}$$

### 4.4 求解：换参数

这是一个非线性 ODE，直接求解需要点技巧。我们引入参数 $\theta$，设：

$$y = a(1 - \cos \theta)$$

> 启发：右边 $\sqrt{1+(y')^2}$ 是弧长元素的几何因子，常用三角代换让它变简单。

计算 $dy$：

$$dy = a \sin \theta \, d\theta$$

回到方程 $y(1 + (y')^2) = 2a$：

$$1 + (y')^2 = \frac{2a}{y} = \frac{2a}{a(1-\cos\theta)} = \frac{2}{1 - \cos\theta}$$

利用 $1 - \cos\theta = 2\sin^2(\theta/2)$，所以：

$$1 + (y')^2 = \frac{1}{\sin^2(\theta/2)}$$

即：

$$(y')^2 = \frac{1}{\sin^2(\theta/2)} - 1 = \frac{1 - \sin^2(\theta/2)}{\sin^2(\theta/2)} = \frac{\cos^2(\theta/2)}{\sin^2(\theta/2)}$$

$$y' = \pm \frac{\cos(\theta/2)}{\sin(\theta/2)} = \pm \cot(\theta/2)$$

取正号（质点在 $\theta$ 增加时下落，$y$ 增加，$x$ 也增加）：

$$y' = \cot(\theta/2)$$

又 $y' = \dfrac{dy/d\theta}{dx/d\theta}$，所以：

$$\frac{dx}{d\theta} = \frac{dy/d\theta}{y'} = \frac{a \sin\theta}{\cot(\theta/2)} = a \sin\theta \cdot \tan(\theta/2)$$

用倍角公式 $\sin\theta = 2 \sin(\theta/2)\cos(\theta/2)$：

$$\frac{dx}{d\theta} = a \cdot 2\sin(\theta/2)\cos(\theta/2) \cdot \frac{\sin(\theta/2)}{\cos(\theta/2)} = 2a \sin^2(\theta/2) = a(1 - \cos\theta)$$

> 注意：$dx/d\theta$ 恰好等于 $dy/d\theta$ 在 $x$ 上的对应物——这给出一个非常对称的结果。

积分，取起点 $\theta=0$ 时 $x=0, y=0$：

$$x(\theta) = \int_0^\theta a(1 - \cos u) \, du = a(\theta - \sin \theta)$$

$$y(\theta) = a(1 - \cos \theta)$$

### 4.5 得到摆线方程

$$\boxed{\begin{cases} x(\theta) = a(\theta - \sin \theta) \\ y(\theta) = a(1 - \cos \theta) \end{cases}} \tag{cycloid}$$

其中 $a > 0$ 是由起点 $(0,0)$ 和终点 $(X, Y)$ 唯一确定的常数（让 $x=a\theta = X, y = 2a = Y$ 时曲线通过终点）。

> 这是经典的**摆线（cycloid）**方程。常数 $a$ 既是"轮子半径"，也是方程中的特征长度。

---

## 5. 摆线是什么：一个轮子的故事

### 5.1 几何构造

想象一个半径为 $a$ 的圆，沿着一条直线**无滑动地滚动**。取轮缘上的一个点（开始时正好接触地面），画出它在不同时间的轨迹，就是一条摆线。

参数方程的物理含义：

- $\theta$：轮子转过的角度。
- $a\theta$：轮子中心水平走过的距离（圆周 = $a\theta$，也等于滚动过的水平距离）。
- $a\sin\theta$：点在轮子上的水平位置（相对中心）。
- $a(1-\cos\theta)$：点在轮子上的竖直位置（相对地面）。当 $\theta=0$，点在地面上 $y=0$；当 $\theta=\pi$，点在最高 $y=2a$；当 $\theta=2\pi$，点又回到地面。

所以水平方向 $x = $ "轮子走了多远" $-$ "点在轮子上的水平偏移"，刚好抵消使轨迹来回"摆"。

### 5.2 一拱

参数 $\theta$ 从 0 到 $2\pi$，对应轮子刚好转一圈，轨迹是一个**摆线拱（cycloid arch）**，长 $2\pi a$，最高 $2a$。

- 起拱点：$(0, 0)$，切线垂直（$dy/dx = \cot(\theta/2)$ 在 $\theta\to 0$ 时趋于 $\infty$）。
- 拱顶：$(\pi a, 2a)$，切线水平。
- 落拱点：$(2\pi a, 0)$，切线又垂直。

> 这个"起落处陡、中段平"的形状，恰恰就是最优滑道的形状。物理直觉和数学严格结果吻合。

### 5.3 看图

下图展示了轮子滚动过程中，轮缘点（红点）描出的轨迹。每个蓝色圆是一个瞬间的轮子位置；红色虚线是圆心到轮缘点的半径。浅灰色曲线是完整摆线。

![摆线的几何构造：轮子滚动](figure2.png)

### 5.4 一拱的几何一览

下图标注了摆线一拱的关键点（起点、四个等分点、终点）和关键长度（半径 $a$）。

![摆线一拱的关键几何](figure4.png)

---

## 6. 摆线一拱的弧长（0 到 2π）

### 6.1 公式

参数曲线的弧长公式：

$$s = \int_\alpha^\beta \sqrt{\left(\frac{dx}{d\theta}\right)^2 + \left(\frac{dy}{d\theta}\right)^2} \, d\theta$$

### 6.2 代入

对摆线：

$$\frac{dx}{d\theta} = a(1 - \cos\theta), \quad \frac{dy}{d\theta} = a \sin\theta$$

$$\left(\frac{dx}{d\theta}\right)^2 + \left(\frac{dy}{d\theta}\right)^2 = a^2(1 - \cos\theta)^2 + a^2 \sin^2\theta$$

展开 $(1 - \cos\theta)^2 = 1 - 2\cos\theta + \cos^2\theta$，与 $\sin^2\theta$ 相加：

$$1 - 2\cos\theta + \cos^2\theta + \sin^2\theta = 1 - 2\cos\theta + 1 = 2(1 - \cos\theta)$$

所以：

$$\sqrt{\left(\frac{dx}{d\theta}\right)^2 + \left(\frac{dy}{d\theta}\right)^2} = a\sqrt{2(1-\cos\theta)}$$

### 6.3 半角公式

$$1 - \cos\theta = 2\sin^2(\theta/2)$$

$$\sqrt{2(1-\cos\theta)} = \sqrt{4\sin^2(\theta/2)} = 2|\sin(\theta/2)|$$

在 $[0, 2\pi]$ 上，$\sin(\theta/2) \geq 0$，所以绝对值可以去掉：

$$\sqrt{\left(\frac{dx}{d\theta}\right)^2 + \left(\frac{dy}{d\theta}\right)^2} = 2a \sin(\theta/2)$$

### 6.4 积分

$$s = \int_0^{2\pi} 2a \sin(\theta/2) \, d\theta$$

$$= 2a \left[ -2\cos(\theta/2) \right]_0^{2\pi}$$

$$= 2a \left[ -2\cos(\pi) - (-2\cos 0) \right]$$

$$= 2a \left[ -2(-1) + 2(1) \right] = 2a \cdot 4 = 8a$$

### 6.5 结果

$$\boxed{s_{\text{arch}} = 8a}$$

> 一拱摆线的弧长是轮子直径 $2a$ 的**四倍**——一个非常干净的结果。

### 6.6 看图

下图展示了弧长元素 $ds$ 沿摆线的累积过程：红色短线段是切向量，把它们"首尾相接"的长度就是总弧长 $8a$。

![弧长元素 ds 沿曲线累积](figure5.png)

---

## 7. 摆线一拱下方的面积

### 7.1 公式

参数曲线下方的面积（这里"下方"指 $y$ 一侧，因为 $y \geq 0$，$x$ 是水平坐标）：

$$A = \int y \, dx$$

参数化下：

$$A = \int_\alpha^\beta y(\theta) \cdot \frac{dx}{d\theta} \, d\theta$$

### 7.2 代入

$$A = \int_0^{2\pi} a(1 - \cos\theta) \cdot a(1 - \cos\theta) \, d\theta = a^2 \int_0^{2\pi} (1 - \cos\theta)^2 \, d\theta$$

### 7.3 展开

$$(1 - \cos\theta)^2 = 1 - 2\cos\theta + \cos^2\theta$$

把 $\cos^2\theta$ 用半角公式 $\cos^2\theta = \dfrac{1 + \cos 2\theta}{2}$：

$$(1-\cos\theta)^2 = 1 - 2\cos\theta + \frac{1}{2} + \frac{\cos 2\theta}{2} = \frac{3}{2} - 2\cos\theta + \frac{\cos 2\theta}{2}$$

### 7.4 逐项积分

在 $[0, 2\pi]$ 上：

- $\int_0^{2\pi} \frac{3}{2} \, d\theta = \frac{3}{2} \cdot 2\pi = 3\pi$
- $\int_0^{2\pi} -2\cos\theta \, d\theta = -2 \cdot [\sin\theta]_0^{2\pi} = 0$
- $\int_0^{2\pi} \frac{\cos 2\theta}{2} \, d\theta = \frac{1}{2} \cdot [\frac{\sin 2\theta}{2}]_0^{2\pi} = 0$

三项相加：$3\pi$。

### 7.5 结果

$$\boxed{A_{\text{arch}} = 3\pi a^2}$$

> 摆线一拱下的面积是 $\pi a^2$（轮子面积）的**三倍**——又是非常干净的结果。

### 7.6 看图

下图展示了面积元素 $y \, dx$ 沿 x 轴的累积过程：把摆线一拱所围区域切成许多竖直窄条（红色矩形），每条面积 $y \, dx$，加总得到 $3\pi a^2$。

![面积元素 y dx 沿 x 累积](figure6.png)

---

## 8. 总结表

| 量 | 公式 | 数值（$a=1$） |
|---|---|---|
| 参数方程 | $x = a(\theta - \sin\theta)$, $y = a(1-\cos\theta)$ | — |
| 一拱水平跨度 | $x(2\pi) = 2\pi a$ | $2\pi \approx 6.28$ |
| 一拱最大高度 | $y(\pi) = 2a$ | $2$ |
| 一拱弧长 | $8a$ | $8$ |
| 一拱面积 | $3\pi a^2$ | $3\pi \approx 9.42$ |
| 轮子面积 | $\pi a^2$ | $\pi \approx 3.14$ |

弧长是直径的四倍，拱面积是轮面积的三倍——简洁到让人怀疑上帝也是工程师。

---

## 9. 复盘：我们做了什么

让我们回头梳理一遍逻辑链：

1. **物理建模**：用能量守恒把速度用高度表示 $v = \sqrt{2gy}$，把时间写成一个泛函 $T[y] = \int \sqrt{(1+(y')^2)/(2gy)} dx$。
2. **变分法**：把"求泛函极值"转化为 E-L 方程 $\partial L/\partial y - d/dx(\partial L/\partial y') = 0$。
3. **Beltrami 恒等式**：因为 $L$ 不显含 $x$，用第一积分简化，得到 $y(1+(y')^2) = 2a$。
4. **参数化**：用三角代换解出 $x(\theta) = a(\theta - \sin\theta), y(\theta) = a(1-\cos\theta)$——摆线方程。
5. **弧长和面积**：用参数弧长公式和半角公式，得到 $8a$ 和 $3\pi a^2$。

每一步都不神秘——只要你愿意跟着算。

---

## 10. 拓展阅读

- 历史背景：约翰·伯努利的原始论文与挑战，牛顿匿名投稿的解答（编辑猜到了作者）。
- 进阶：从最速降线推广到"等时曲线（tautochrone）"——摆线不仅是最快的，还是**不管从哪点出发都同时到达**的曲线。伽利略误以为是圆弧，摆线才是对的。
- 物理实现：把小球放在最速降线形状的滑槽里释放，与直线轨道对比时间。可以用 3D 打印做一个。

---

## 附录：如何重新生成配图

如果修改了脚本或者想重新生成图片，在当前目录运行：

```bash
python figures.py all      # 重新生成全部 6 张图
python figures.py verify   # 数值验证 8a 和 3πa²
python figures.py N        # 只生成第 N 张（N=1~6）
```

生成的图片会保存到当前目录，文件名 `figure1.png` 到 `figure6.png`，直接被上面的 Markdown 引用。

