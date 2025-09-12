# The Little Book of Linear Algebra
线性代数小册子

A concise, beginner-friendly introduction to the core ideas of linear algebra.
简洁、适合初学者的线性代数核心思想介绍。

## Formats
格式

*   [Download PDF](book.pdf) – print-ready version
    [下载 PDF](book.pdf) – 可打印版本
*   [Download EPUB](book.epub) – e-reader friendly
    [下载 EPUB](book.epub) – 适合电子阅读器
*   [View LaTeX](book.tex) – Latex source
    [查看 LaTeX](book.tex) – Latex 源

# Chapter 1. Vectors
第 1 章 向量

## 1.1 Scalars and Vectors
1.1 标量和矢量

A scalar is a single numerical quantity, most often taken from the real numbers, denoted by $\mathbb{R}$. Scalars are the fundamental building blocks of arithmetic: they can be added, subtracted, multiplied, and, except in the case of zero, divided. In linear algebra, scalars play the role of coefficients, scaling factors, and entries of larger structures such as vectors and matrices. They provide the weights by which more complex objects are measured and combined. A vector is an ordered collection of scalars, arranged either in a row or a column. When the scalars are real numbers, the vector is said to belong to *real* $n$\-dimensional space, written
标量是一个单一的数值，通常取自实数，用 $\mathbb{R}$ 表示。标量是算术的基本组成部分：它们可以进行加、减、乘和除（零除外）。在线性代数中，标量充当系数、比例因子以及向量和矩阵等更大结构中的元素。它们提供权重，用于测量和组合更复杂的对象。向量是按行或列排列的标量的有序集合。当标量为实数时，该向量被称为属于*实* $n$ 维空间，写为

$$
\mathbb{R}^n = \{ (x_1, x_2, \dots, x_n) \mid x_i \in \mathbb{R} \}.
$$

An element of $\mathbb{R}^n$ is called a vector of dimension $n$ or an *n*\-vector. The number $n$ is called the dimension of the vector space. Thus $\mathbb{R}^2$ is the space of all ordered pairs of real numbers, $\mathbb{R}^3$ the space of all ordered triples, and so on.
$\mathbb{R}^n$ 中的一个元素称为维度为 $n$ 的向量或 *n* 向量。数字 $n$ 称为向量空间的维数。因此， $\mathbb{R}^2$ 是所有有序实数对的空间， $\mathbb{R}^3$ 是所有有序三元组的空间，等等。

Example 1.1.1.
例 1.1.1。

*   A 2-dimensional vector: $(3, -1) \in \mathbb{R}^2$.
    二维向量： $(3, -1) \in \mathbb{R}^2$ 。
*   A 3-dimensional vector: $(2, 0, 5) \in \mathbb{R}^3$.
    三维向量： $(2, 0, 5) \in \mathbb{R}^3$ 。
*   A 1-dimensional vector: $(7) \in \mathbb{R}^1$, which corresponds to the scalar 7 itself.
    一维向量： $(7) \in \mathbb{R}^1$ ，对应于标量 7 本身。

Vectors are often written vertically in column form, which emphasizes their role in matrix multiplication:
向量通常以列的形式垂直书写，这强调了它们在矩阵乘法中的作用：

$$
\mathbf{v} = \begin{bmatrix}2 \\0 \\5 \end{bmatrix} \in \mathbb{R}^3.
$$

The vertical layout makes the structure clearer when we consider linear combinations or multiply matrices by vectors.
当我们考虑线性组合或矩阵乘以向量时，垂直布局使结构更加清晰。

### Geometric Interpretation
几何解释

In $\mathbb{R}^2$, a vector $(x_1, x_2)$ can be visualized as an arrow starting at the origin $(0,0)$ and ending at the point $(x_1, x_2)$. Its length corresponds to the distance from the origin, and its orientation gives a direction in the plane. In $\mathbb{R}^3$, the same picture extends into three dimensions: a vector is an arrow from the origin to $(x_1, x_2, x_3)$. Beyond three dimensions, direct visualization is no longer possible, but the algebraic rules of vectors remain identical. Even though we cannot draw a vector in $\mathbb{R}^{10}$, it behaves under addition, scaling, and transformation exactly as a 2- or 3-dimensional vector does. This abstract point of view is what allows linear algebra to apply to data science, physics, and machine learning, where data often lives in very high-dimensional spaces. Thus a vector may be regarded in three complementary ways:
在 $\mathbb{R}^2$ 中，向量 $(x_1, x_2)$ 可以可视化为一个从原点 $(0,0)$ 开始到点 $(x_1, x_2)$ 结束的箭头。它的长度对应于与原点的距离，其方向给出了平面内的方向。在 $\mathbb{R}^3$ 中，同样的图像延伸到三维空间：向量是一个从原点指向 $(x_1, x_2, x_3)$ 的箭头。超过三维空间后，直接可视化就不再可能，但向量的代数规则保持不变。即使我们无法在 $\mathbb{R}^{10}$ 中绘制向量，它在加法、缩放和变换下的行为与二维或三维向量完全相同。这种抽象的观点使得线性代数能够应用于数据科学、物理学和机器学习，这些领域的数据通常存在于非​​常高维的空间中。因此，向量可以从三个互补的角度来看待：

1.  As a point in space, described by its coordinates.
    作为空间中的一个点，由其坐标描述。
2.  As a displacement or arrow, described by a direction and a length.
    作为位移或箭头，由方向和长度描述。
3.  As an abstract element of a vector space, whose properties follow algebraic rules independent of geometry.
    作为向量空间的抽象元素，其属性遵循与几何无关的代数规则。

### Notation
符号

*   Vectors are written in boldface lowercase letters: $\mathbf{v}, \mathbf{w}, \mathbf{x}$.
    向量以粗体小写字母表示： $\mathbf{v}, \mathbf{w}, \mathbf{x}$ 。
*   The *i*\-th entry of a vector $\mathbf{v}$ is written $v_i$, where indices begin at 1.
    向量 $\mathbf{v}$ 的第 - 个元素写为 𝑣 𝑖 v i ​ ，其中索引从 1 开始。
*   The set of all *n*\-dimensional vectors over $\mathbb{R}$ is denoted $\mathbb{R}^n$.
    $\mathbb{R}$ 上的所有 *n* 维向量的集合记为 $\mathbb{R}^n$ 。
*   Column vectors will be the default form unless otherwise stated.
    除非另有说明，列向量将是默认形式。

### Why begin here?
为什么从这里开始？

Scalars and vectors form the atoms of linear algebra. Every structure we will build-vector spaces, linear transformations, matrices, eigenvalues-relies on the basic notions of number and ordered collection of numbers. Once vectors are understood, we can define operations such as addition and scalar multiplication, then generalize to subspaces, bases, and coordinate systems. Eventually, this framework grows into the full theory of linear algebra, with powerful applications to geometry, computation, and data.
标量和向量构成了线性代数的原子。我们将要构建的每一个结构——向量空间、线性变换、矩阵、特征值——都依赖于数和有序数集的基本概念。一旦理解了向量，我们就可以定义诸如加法和标量乘法之类的运算，然后推广到子空间、基和坐标系。最终，这个框架将发展成为完整的线性代数理论，并在几何、计算和数据领域拥有强大的应用。

### Exercises 1.1
练习 1.1

1.  Write three different vectors in $\mathbb{R}^2$ and sketch them as arrows from the origin. Identify their coordinates explicitly.
    在 $\mathbb{R}^2$ 中写出三个不同的向量，并将它们画成从原点出发的箭头。明确指出它们的坐标。
2.  Give an example of a vector in $\mathbb{R}^4$. Can you visualize it directly? Explain why high-dimensional visualization is challenging.
    给出 $\mathbb{R}^4$ 中一个向量的例子。你能直接将其可视化吗？解释为什么高维可视化具有挑战性。
3.  Let $\mathbf{v} = (4, -3, 2)$. Write $\mathbf{v}$ in column form and state $v_1, v_2, v_3$.
    令 $\mathbf{v} = (4, -3, 2)$ 。将 $\mathbf{v}$ 写成列形式，并说明 𝑣 1 , 𝑣 2 , 𝑣 3 v 1 ​ ，v 2 ​ ，v 3 ​ .
4.  In what sense is the set $\mathbb{R}^1$ both a line and a vector space? Illustrate with examples.
    在什么意义上集合 $\mathbb{R}^1$ 既是线空间又是向量空间？请举例说明。
5.  Consider the vector $\mathbf{u} = (1,1,\dots,1) \in \mathbb{R}^n$. What is special about this vector when $n$ is large? What might it represent in applications?
    考虑向量 $\mathbf{u} = (1,1,\dots,1) \in \mathbb{R}^n$ 。当 $n$ 很大时，这个向量有什么特殊之处？它在应用中可能代表什么？

## 1.2 Vector Addition and Scalar Multiplication
1.2 向量加法和标量乘法

Vectors in linear algebra are not static objects; their power comes from the operations we can perform on them. Two fundamental operations define the structure of vector spaces: addition and scalar multiplication. These operations satisfy simple but far-reaching rules that underpin the entire subject.
线性代数中的向量并非静态对象；它们的力量源于我们可以对它们执行的运算。两个基本运算定义了向量空间的结构：加法和标量乘法。这两个运算满足一些简单却影响深远的规则，这些规则构成了整个线性代数学科的基础。

### Vector Addition
向量加法

Given two vectors of the same dimension, their sum is obtained by adding corresponding entries. Formally, if
给定两个相同维度的向量，它们的和可以通过添加相应的元素来获得。形式上，如果

$$
\mathbf{u} = (u_1, u_2, \dots, u_n), \quad\mathbf{v} = (v_1, v_2, \dots, v_n),
$$

then their sum is
那么它们的总和是

$$
\mathbf{u} + \mathbf{v} = (u_1+v_1, u_2+v_2, \dots, u_n+v_n).
$$

Example 1.2.1. Let $\mathbf{u} = (2, -1, 3)$ and $\mathbf{v} = (4, 0, -5)$. Then
例 1.2.1。 设 $\mathbf{u} = (2, -1, 3)$ 和 $\mathbf{v} = (4, 0, -5)$ 。则

$$
\mathbf{u} + \mathbf{v} = (2+4, -1+0, 3+(-5)) = (6, -1, -2).
$$

Geometrically, vector addition corresponds to the *parallelogram rule*. If we draw both vectors as arrows from the origin, then placing the tail of one vector at the head of the other produces the sum. The diagonal of the parallelogram they form represents the resulting vector.
从几何学上讲，向量加法对应于*平行四边形法则* 。如果我们将两个向量都画成从原点出发的箭头，那么将一个向量的尾部放在另一个向量的头部，就能得到向量的和。它们构成的平行四边形的对角线代表最终的向量。

### Scalar Multiplication
标量乘法

Multiplying a vector by a scalar stretches or shrinks the vector while preserving its direction, unless the scalar is negative, in which case the vector is also reversed. If $c \in \mathbb{R}$ and
将矢量乘以标量会拉伸或收缩矢量，同时保持其方向，除非标量 负数，在这种情况下向量也会反转。如果 $c \in \mathbb{R}$ 和

$$
\mathbf{v} = (v_1, v_2, \dots, v_n),
$$

then
然后

$$
c \mathbf{v} = (c v_1, c v_2, \dots, c v_n).
$$

Example 1.2.2. Let $\mathbf{v} = (3, -2)$ and $c = -2$. Then
例 1.2.2。 设 $\mathbf{v} = (3, -2)$ 和 $c = -2$ 。则

$$
c\mathbf{v} = -2(3, -2) = (-6, 4).
$$

This corresponds to flipping the vector through the origin and doubling its length.
这相当于通过原点翻转向量并使其长度加倍。

### Linear Combinations
线性组合

The interaction of addition and scalar multiplication allows us to form *linear combinations*. A linear combination of vectors $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k$ is any vector of the form
加法和标量乘法的相互作用使我们能够形成*线性组合* 。向量𝑣的线性组合 1 , 𝑣 2 , … , 𝑣 𝑘 v 1 ​ ，v 2 ​ ，…，v k ​ 是任意形式的向量

$$
c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k, \quad c_i \in \mathbb{R}.
$$

Linear combinations are the mechanism by which we generate new vectors from existing ones. The span of a set of vectors-the collection of all their linear combinations-will later lead us to the idea of a subspace.
线性组合是一种从现有向量生成新向量的机制。一组向量的跨度——它们所有线性组合的集合——稍后会引出子空间的概念。

Example 1.2.3. Let $\mathbf{v}_1 = (1,0)$ and $\mathbf{v}_2 = (0,1)$. Then any vector $(a,b)\in\mathbb{R}^2$ can be expressed as
例 1.2.3。 设 $\mathbf{v}_1 = (1,0)$ 和 $\mathbf{v}_2 = (0,1)$ 。则任意向量 $(a,b)\in\mathbb{R}^2$ 可以表示为

$$
a\mathbf{v}_1 + b\mathbf{v}_2.
$$

Thus $(1,0)$ and $(0,1)$ form the basic building blocks of the plane.
因此 $(1,0)$ 和 $(0,1)$ 构成了平面的基本构造块。

### Notation
符号

*   Addition: $\mathbf{u} + \mathbf{v}$ means component-wise addition.
    加法： $\mathbf{u} + \mathbf{v}$ 表示逐个组件的加法。
*   Scalar multiplication: $c\mathbf{v}$ scales each entry of $\mathbf{v}$ by $c$.
    标量乘法： $c\mathbf{v}$ 将 $\mathbf{v}$ 的每个条目乘以 $c$ 。
*   Linear combination: a sum of the form $c_1 \mathbf{v}_1 + \cdots + c_k \mathbf{v}_k$.
    线性组合：𝑐 形式的和 1 𝑣 1 + ⋯ + 𝑐 𝑘 𝑣 𝑘 c 1 ​ v 1 ​ +⋯+c k ​ v k ​ .

### Why this matters
为什么这很重要

Vector addition and scalar multiplication are the defining operations of linear algebra. They give structure to vector spaces, allow us to describe geometric phenomena like translation and scaling, and provide the foundation for solving systems of equations. Everything that follows-basis, dimension, transformations-builds on these simple but profound rules.
向量加法和标量乘法是线性代数的定义运算。它们赋予向量空间结构，使我们能够描述平移和缩放等几何现象，并为方程组的求解奠定基础。之后的一切——基、维度、变换——都建立在这些简单而深刻的规则之上。

### Exercises 1.2
练习 1.2

1.  Compute $\mathbf{u} + \mathbf{v}$ where $\mathbf{u} = (1,2,3)$ and $\mathbf{v} = (4, -1, 0)$.
    计算 $\mathbf{u} + \mathbf{v}$ ，其中 $\mathbf{u} = (1,2,3)$ 和 $\mathbf{v} = (4, -1, 0)$ 。
2.  Find $3\\mathbf{v}$where$\\mathbf{v} = (-2,5)$. Sketch both vectors to illustrate the scaling.
    求 $3\\mathbf{v} $where$ \\mathbf{v} = (-2,5)$。画出两个向量的示意图，以说明缩放关系。
3.  Show that $(5,7)$ can be written as a linear combination of $(1,0)$ and $(0,1)$.
    证明 $(5,7)$ 可以写成 $(1,0)$ 和 $(0,1)$ 的线性组合。
4.  Write $(4,4)$ as a linear combination of $(1,1)$ and $(1,-1)$.
    将 $(4,4)$ 写为 $(1,1)$ 和 $(1,-1)$ 的线性组合。
5.  Prove that if $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$, then $(c+d)(\mathbf{u}+\mathbf{v}) = c\mathbf{u} + c\mathbf{v} + d\mathbf{u} + d\mathbf{v}$ for scalars $c,d \in \mathbb{R}$.
    证明如果 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ ，则对于标量 $c,d \in \mathbb{R}$ 有 $(c+d)(\mathbf{u}+\mathbf{v}) = c\mathbf{u} + c\mathbf{v} + d\mathbf{u} + d\mathbf{v}$ 。

## 1.3 Dot Product, Norms, and Angles
1.3 点积、范数和角

The dot product is the fundamental operation that links algebra and geometry in vector spaces. It allows us to measure lengths, compute angles, and determine orthogonality. From this single definition flow the notions of *norm* and *angle*, which give geometry to abstract vector spaces.
点积是向量空间中连接代数和几何的基本运算。它使我们能够测量长度、计算角度并确定正交性。从这个单一定义中衍生出*范*数和 *角度* ，它为抽象向量空间提供几何形状。

### The Dot Product
点积

For two vectors in $\mathbb{R}^n$, the dot product (also called the inner product) is defined by
对于 $\mathbb{R}^n$ 中的两个向量，点积（也称为内积）定义为

$$
\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n.
$$

Equivalently, in matrix notation:
等效地，用矩阵表示法表示：

$$
\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v}.
$$

Example 1.3.1. Let $\mathbf{u} = (2, -1, 3)$ and $\mathbf{v} = (4, 0, -2)$. Then
例 1.3.1。 设 $\mathbf{u} = (2, -1, 3)$ 和 $\mathbf{v} = (4, 0, -2)$ 。则

$$
\mathbf{u} \cdot \mathbf{v} = 2\cdot 4 + (-1)\cdot 0 + 3\cdot (-2) = 8 - 6 = 2.
$$

The dot product outputs a single scalar, not another vector.
点积输出单个标量，而不是另一个向量。

### Norms (Length of a Vector)
范数（向量的长度）

The *Euclidean norm* of a vector is the square root of its dot product with itself:
向量的*欧几里得范数*是其与自身的点积的平方根：

$$
\|\mathbf{v}\| = \sqrt{\mathbf{v} \cdot \mathbf{v}} = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}.
$$

This generalizes the Pythagorean theorem to arbitrary dimensions.
这将勾股定理推广到任意维度。

Example 1.3.2. For $\mathbf{v} = (3, 4)$,
例 1.3.2。 对于 $\mathbf{v} = (3, 4)$ ，

$$
\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5.
$$

This is exactly the length of the vector as an arrow in the plane.
这正是平面中箭头所指的矢量的长度。

### Angles Between Vectors
向量之间的角度

The dot product also encodes the angle between two vectors. For nonzero vectors $\mathbf{u}, \mathbf{v}$,
点积也编码了两个向量之间的角度。对于非零向量 $\mathbf{u}, \mathbf{v}$ ，

$$
\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \, \|\mathbf{v}\| \cos \theta,
$$

where $\theta$ is the angle between them. Thus,
其中 $\theta$ 是它们之间的角度。因此，

$$
\cos \theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}.
$$

Example 1.3.3. Let $\mathbf{u} = (1,0)$ and $\mathbf{v} = (0,1)$. Then
例 1.3.3。 设 $\mathbf{u} = (1,0)$ 和 $\mathbf{v} = (0,1)$ 。则

$$
\mathbf{u} \cdot \mathbf{v} = 0, \quad \|\mathbf{u}\| = 1, \quad \|\mathbf{v}\| = 1.
$$

Hence
因此

$$
\cos \theta = \frac{0}{1\cdot 1} = 0 \quad \Rightarrow \quad \theta = \frac{\pi}{2}.
$$

The vectors are perpendicular.
这些向量是垂直的。

### Orthogonality
正交性

Two vectors are said to be orthogonal if their dot product is zero:
如果两个向量的点积为零，则称它们正交：

$$
\mathbf{u} \cdot \mathbf{v} = 0.
$$

Orthogonality generalizes the idea of perpendicularity from geometry to higher dimensions.
正交性将垂直性的概念从几何学推广到更高维度。

### Notation
符号

*   Dot product: $\mathbf{u} \cdot \mathbf{v}$.
    点积： $\mathbf{u} \cdot \mathbf{v}$ 。
*   Norm (length): $|\mathbf{v}|$.
    规范（长度）： $|\mathbf{v}|$ 。
*   Orthogonality: $\mathbf{u} \perp \mathbf{v}$ if $\mathbf{u} \cdot \mathbf{v} = 0$.
    正交性：如果为 $\mathbf{u} \cdot \mathbf{v} = 0$ ，则为 $\mathbf{u} \perp \mathbf{v}$ 。

### Why this matters
为什么这很重要

The dot product turns vector spaces into geometric objects: vectors gain lengths, angles, and notions of perpendicularity. This foundation will later support the study of orthogonal projections, Gram–Schmidt orthogonalization, eigenvectors, and least squares problems.
点积将向量空间转化为几何对象：向量获得长度、角度和垂直度的概念。这一基础将为后续的正交投影、格拉姆-施密特正交化、特征向量和最小二乘问题的研究奠定基础。

### Exercises 1.3
练习 1.3

1.  Compute $\mathbf{u} \cdot \mathbf{v}$ for $\mathbf{u} = (1,2,3)$, $\mathbf{v} = (4,5,6)$.
    计算 $\mathbf{u} = (1,2,3)$ 、 $\mathbf{v} = (4,5,6)$ 的 $\mathbf{u} \cdot \mathbf{v}$ 。
2.  Find the norm of $\mathbf{v} = (2, -2, 1)$.
    求出 $\mathbf{v} = (2, -2, 1)$ 的范数。
3.  Determine whether $\mathbf{u} = (1,1,0)$ and $\mathbf{v} = (1,-1,2)$ are orthogonal.
    确定 $\mathbf{u} = (1,1,0)$ 和 $\mathbf{v} = (1,-1,2)$ 是否正交。
4.  Let $\mathbf{u} = (3,4)$, $\mathbf{v} = (4,3)$. Compute the angle between them.
    令 $\mathbf{u} = (3,4)$ , $\mathbf{v} = (4,3)$ 。计算它们之间的角度。
5.  Prove that $|\mathbf{u} + \mathbf{v}|^2 = |\mathbf{u}|^2 + |\mathbf{v}|^2 + 2\mathbf{u}\cdot \mathbf{v}$. This identity is the algebraic version of the Law of Cosines.
    证明 $|\mathbf{u} + \mathbf{v}|^2 = |\mathbf{u}|^2 + |\mathbf{v}|^2 + 2\mathbf{u}\cdot \mathbf{v}$ 。这个恒等式是余弦定理的代数形式。

## 1.4 Orthogonality
1.4 正交性

Orthogonality captures the notion of perpendicularity in vector spaces. It is one of the most important geometric ideas in linear algebra, allowing us to decompose vectors, define projections, and construct special bases with elegant properties.
正交性捕捉了向量空间中垂直性的概念。它是线性代数中最重要的几何概念之一，它使我们能够分解向量、定义投影，并构造具有优雅性质的特殊基。

### Definition
定义

Two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ are said to be orthogonal if their dot product is zero:
如果两个向量 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ 的点积为零，则称它们正交：

$$
\mathbf{u} \cdot \mathbf{v} = 0.
$$

This condition ensures that the angle between them is $\pi/2$ radians (90 degrees).
此条件确保它们之间的角度为 $\pi/2$ 弧度（90 度）。

Example 1.4.1. In $\mathbb{R}^2$, the vectors $(1,2)$ and $(2,-1)$ are orthogonal since
例 1.4.1。 在 $\mathbb{R}^2$ 中，向量 $(1,2)$ 和 $(2,-1)$ 是正交的，因为

$$
(1,2) \cdot (2,-1) = 1\cdot 2 + 2\cdot (-1) = 0.
$$

### Orthogonal Sets
正交集

A collection of vectors is called orthogonal if every distinct pair of vectors in the set is orthogonal. If, in addition, each vector has norm 1, the set is called orthonormal.
如果一组向量中每对不同的向量都是正交的，则称该集合为正交向量。此外，如果每个向量的范数均为 1，则该集合称为标准正交向量集。

Example 1.4.2. In $\mathbb{R}^3$, the standard basis vectors
例 1.4.2。 在 $\mathbb{R}^3$ 中，标准基向量

$$
\mathbf{e}_1 = (1,0,0), \quad \mathbf{e}_2 = (0,1,0), \quad \mathbf{e}_3 = (0,0,1)
$$

form an orthonormal set: each has length 1, and their dot products vanish when the indices differ.
形成一个正交集：每个集的长度为 1，并且当索引不同时，它们的点积消失。

### Projections
预测

Orthogonality makes possible the decomposition of a vector into two components: one parallel to another vector, and one orthogonal to it. Given a nonzero vector $\mathbf{u}$ and any vector $\mathbf{v}$, the projection of $\mathbf{v}$ onto $\mathbf{u}$ is
正交性使得将一个向量分解为两个分量成为可能：一个与另一个向量平行，另一个 与其正交。给定一个非零向量 $\mathbf{u}$ 和任意向量 $\mathbf{v}$ ，则 $\mathbf{v}$ 的投影 到 $\mathbf{u}$ 是

$$
\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u}.
$$

The difference
区别

$$
\mathbf{v} - \text{proj}_{\mathbf{u}}(\mathbf{v})
$$

is orthogonal to $\mathbf{u}$. Thus every vector can be decomposed uniquely into a parallel and perpendicular part with respect to another vector.
与 $\mathbf{u}$ 正交。因此，每个向量都可以唯一地分解为相对于另一个向量平行和垂直的部分。

Example 1.4.3. Let $\mathbf{u} = (1,0)$, $\mathbf{v} = (2,3)$. Then
例 1.4.3。 令 $\mathbf{u} = (1,0)$ ， $\mathbf{v} = (2,3)$ 。然后

$$
\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{(1,0)\cdot(2,3)}{(1,0)\cdot(1,0)} (1,0)= \frac{2}{1}(1,0) = (2,0).
$$

Thus
因此

$$
\mathbf{v} = (2,3) = (2,0) + (0,3),
$$

where $(2,0)$ is parallel to $(1,0)$ and $(0,3)$ is orthogonal to it.
其中 $(2,0)$ 与 $(1,0)$ 平行， $(0,3)$ 与 $(1,0)$ 正交。

### Orthogonal Decomposition
正交分解

In general, if $\mathbf{u} \neq \mathbf{0}$ and $\mathbf{v} \in \mathbb{R}^n$, then
一般来说，如果 $\mathbf{u} \neq \mathbf{0}$ 和 $\mathbf{v} \in \mathbb{R}^n$ ，那么

$$
\mathbf{v} = \text{proj}\_{\mathbf{u}}(\mathbf{v}) + \big(\mathbf{v} - \text{proj}\_{\mathbf{u}}(\mathbf{v})\big),
$$

where the first term is parallel to $\mathbf{u}$ and the second term is orthogonal. This decomposition underlies methods such as least squares approximation and the Gram–Schmidt process.
其中第一项平行于 $\mathbf{u}$ ，第二项正交。这种分解是最小二乘近似和格拉姆-施密特过程等方法的基础。

### Notation
符号

*   $\mathbf{u} \perp \mathbf{v}$: vectors $\mathbf{u}$ and $\mathbf{v}$ are orthogonal.
    $\mathbf{u} \perp \mathbf{v}$ ：向量 $\mathbf{u}$ 和 $\mathbf{v}$ 正交。
*   An orthogonal set: vectors pairwise orthogonal.
    正交集：向量两两正交。
*   An orthonormal set: pairwise orthogonal, each of norm 1.
    正交集：两两正交，每组范数为 1。

### Why this matters
为什么这很重要

Orthogonality gives structure to vector spaces. It provides a way to separate independent directions cleanly, simplify computations, and minimize errors in approximations. Many powerful algorithms in numerical linear algebra and data science (QR decomposition, least squares regression, PCA) rely on orthogonality.
正交性赋予向量空间结构。它提供了一种清晰地分离独立方向、简化计算并最小化近似误差的方法。数值线性代数和数据科学中许多强大的算法（例如 QR 分解、最小二乘回归、主成分分析）都依赖于正交性。

### Exercises 1.4
练习 1.4

1.  Verify that the vectors $(1,2,2)$ and $(2,0,-1)$ are orthogonal.
    验证向量 $(1,2,2)$ 和 $(2,0,-1)$ 是否正交。
2.  Find the projection of $(3,4)$ onto $(1,1)$.
    找到 $(3,4)$ 到 $(1,1)$ 的投影。
3.  Show that any two distinct standard basis vectors in $\mathbb{R}^n$ are orthogonal.
    证明 $\mathbb{R}^n$ 中的任意两个不同的标准基向量都是正交的。
4.  Decompose $(5,2)$ into components parallel and orthogonal to $(2,1)$.
    将 $(5,2)$ 分解为与 $(2,1)$ 平行且正交的分量。
5.  Let $\mathbf{u}, \mathbf{v}$ be orthogonal nonzero vectors. (a) Show that $(\mathbf{u}+\mathbf{v})\cdot(\mathbf{u}-\mathbf{v})=\lVert \mathbf{u}\rVert^2-\lVert \mathbf{v}\rVert^2.$ (b) For what condition on $\mathbf{u}$ and $\mathbf{v}$ does $(\mathbf{u}+\mathbf{v})\cdot(\mathbf{u}-\mathbf{v})=0$?
    令 $\mathbf{u}, \mathbf{v}$ 为正交非零向量。（a）证明 $(\mathbf{u}+\mathbf{v})\cdot(\mathbf{u}-\mathbf{v})=\lVert \mathbf{u}\rVert^2-\lVert \mathbf{v}\rVert^2.$ （b） $(\mathbf{u}+\mathbf{v})\cdot(\mathbf{u}-\mathbf{v})=0$ 对 $\mathbf{u}$ 和 $\mathbf{v}$ 满足什么条件？

# Chapter 2. Matrices
第 2 章矩阵

## 2.1 Definition and Notation
2.1 定义和符号

Matrices are the central objects of linear algebra, providing a compact way to represent and manipulate linear transformations, systems of equations, and structured data. A matrix is a rectangular array of numbers arranged in rows and columns.
矩阵是线性代数的核心对象，它提供了一种简洁的方式来表示和操作线性变换、方程组和结构化数据。矩阵是由按行和列排列的数字组成的矩形阵列。

### Formal Definition
正式定义

An $m \times n$ matrix is an array with $m$ rows and $n$ columns, written
$m \times n$ 矩阵是具有 $m$ 行和 $n$ 列的数组，写为

$$
A =\begin{bmatrix}a_{11} & a_{12} & \cdots & a_{1n} \\a_{21} & a_{22} & \cdots & a_{2n} \\\vdots & \vdots & \ddots & \vdots \\a_{m1} & a_{m2} & \cdots & a_{mn}\end{bmatrix}.
$$

Each entry $a_{ij}$ is a scalar, located in the *i*\-th row and *j*\-th column. The size (or dimension) of the matrix is denoted by $m \times n$.
每个条目𝑎 𝑖 𝑗 a 伊奇 ​ 是一个标量，位于第 - 行和第 - 列。矩阵的大小（或维度）用 $m \times n$ 表示。

*   If $m = n$, the matrix is square.
    如果为 $m = n$ ，则矩阵为方阵。
*   If $m = 1$, the matrix is a row vector.
    如果为 $m = 1$ ，则该矩阵为行向量。
*   If $n = 1$, the matrix is a column vector.
    如果为 $n = 1$ ，则矩阵为列向量。

Thus, vectors are simply special cases of matrices.
因此，向量只是矩阵的特殊情况。

### Examples
示例

Example 2.1.1. A 2×3 matrix:
例 2.1.1. 2×3 矩阵：

$$
A = \begin{bmatrix}1 & -2 & 4 \\0 & 3 & 5\end{bmatrix}.
$$

Here, $a_{12} = -2$, $a_{23} = 5$, and the matrix has 2 rows, 3 columns.
这里， $a_{12} = -2$ ， $a_{23} = 5$ ，矩阵有 2 行，3 列。

Example 2.1.2. A 3×3 square matrix:
例 2.1.2. 3×3 方阵：

$$
B = \begin{bmatrix}2 & 0 & 1 \\-1 & 3 & 4 \\0 & 5 & -2\end{bmatrix}.
$$

This will later serve as the representation of a linear transformation on $\mathbb{R}^3$.
这稍后将作为 $\mathbb{R}^3$ 的线性变换的表示。

### Indexing and Notation
索引和符号

*   Matrices are denoted by uppercase bold letters: $A, B, C$.
    矩阵用大写粗体字母表示： $A, B, C$ 。
*   Entries are written as $a_{ij}$, with the row index first, column index second.
    条目写为𝑎 𝑖 𝑗 a 伊奇 ​ ，其中行索引在前，列索引在后。
*   The set of all real $m \times n$ matrices is denoted $\mathbb{R}^{m \times n}$.
    所有实数 $m \times n$ 矩阵的集合表示为 $\mathbb{R}^{m \times n}$ 。

Thus, a matrix is a function $A: {1,\dots,m} \times {1,\dots,n} \to \mathbb{R}$, assigning a scalar to each row-column position.
因此，矩阵是一个函数 $A: {1,\dots,m} \times {1,\dots,n} \to \mathbb{R}$ ，为每个行列位置分配一个标量。

### Why this matters
为什么这很重要

Matrices generalize vectors and give us a language for describing linear operations systematically. They encode systems of equations, rotations, projections, and transformations of data. With matrices, algebra and geometry come together: a single compact object can represent both numerical data and functional rules.
矩阵推广了向量，并为我们提供了一种系统地描述线性运算的语言。它们对方程组、旋转、投影和数据变换进行编码。矩阵将代数和几何结合在一起：一个紧凑的对象既可以表示数值数据，又可以表示函数规则。

### Exercises 2.1
练习 2.1

1.  Write a $3 \\times 2$matrix of your choice and identify its entries$a\_{ij}$.
    写出 $3 \\times 2 $matrix of your choice and identify its entries$ a\_{ij}$。
2.  Is every vector a matrix? Is every matrix a vector? Explain.
    每个向量都是矩阵吗？每个矩阵都是向量吗？请解释。
3.  Which of the following are square matrices: $A \in \mathbb{R}^{4\times4}$, $B \in \mathbb{R}^{3\times5}$, $C \in \mathbb{R}^{1\times1}$?
    下列哪些是正方形 矩阵： $A \in \mathbb{R}^{4\times4}$ ， $B \in \mathbb{R}^{3\times5}$ ， $C \in \mathbb{R}^{1\times1}$ ？
4.  Let
    让

$$
D = \begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix}
$$

What kind of matrix is this? 5. Consider the matrix
这是什么类型的矩阵？5. 考虑矩阵

$$
E = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

Express $e_{11}, e_{12}, e_{21}, e_{22}$ explicitly.
快递𝑒 11 , 𝑒 12 , 𝑒 21 , 𝑒 22 e 11 ​ ，e 12 ​ ，e 21 ​ ，e 22 ​ 明确地。

## 2.2 Matrix Addition and Multiplication
2.2 矩阵加法和乘法

Once matrices are defined, the next step is to understand how they combine. Just as vectors gain meaning through addition and scalar multiplication, matrices become powerful through two operations: addition and multiplication.
定义好矩阵后，下一步就是理解它们是如何组合的。正如向量通过加法和标量乘法获得意义一样，矩阵也通过两种运算变得强大：加法和乘法。

### Matrix Addition
矩阵加法

Two matrices of the same size are added by adding corresponding entries. If
两个大小相同的矩阵可以通过添加相应的元素来相加。如果

$$
A = [a_{ij}] \in \mathbb{R}^{m \times n}, \quadB = [b_{ij}] \in \mathbb{R}^{m \times n},
$$

then
然后

$$
A + B = [a_{ij} + b_{ij}] \in \mathbb{R}^{m \times n}.
$$

Example 2.2.1. Let
例 2.2.1. 设

$$
A = \begin{bmatrix}1 & 2 \\3 & 4\end{bmatrix}, \quadB = \begin{bmatrix}-1 & 0 \\5 & 2\end{bmatrix}.
$$

Then
然后

$$
A + B = \begin{bmatrix}1 + (-1) & 2 + 0 \\3 + 5 & 4 + 2\end{bmatrix} =\begin{bmatrix}0 & 2 \\8 & 6\end{bmatrix}.
$$

Matrix addition is commutative ($A+B = B+A$) and associative ($(A+B)+C = A+(B+C)$). The zero matrix, with all entries 0, acts as the additive identity.
矩阵加法满足交换律 ( $A+B = B+A$ ) 和结合律 ( $(A+B)+C = A+(B+C)$ )。零矩阵（所有元素均为 0）充当加法恒等式。

### Scalar Multiplication
标量乘法

For a scalar $c \in \mathbb{R}$ and a matrix $A = [[a_{ij}]$, we define
对于标量 $c \in \mathbb{R}$ 和矩阵 $A = [[a_{ij}]$ ，我们定义

$$
cA = [c \cdot a_{ij}].
$$

This stretches or shrinks all entries of the matrix uniformly.
这会均匀地拉伸或收缩矩阵的所有条目。

Example 2.2.2. If
例 2.2.2. 如果

$$
A = \begin{bmatrix}2 & -1 \\0 & 3\end{bmatrix}, \quad c = -2,
$$

then
然后

$$
cA = \begin{bmatrix}-4 & 2 \\0 & -6\end{bmatrix}.
$$

### Matrix Multiplication
矩阵乘法

The defining operation of matrices is multiplication. If
矩阵的定义运算是乘法。如果

$$
A \in \mathbb{R}^{m \times n}, \quad B \in \mathbb{R}^{n \times p},
$$

then their product is the $m \times p$ matrix
那么它们的乘积就是 $m \times p$ 矩阵

$$
AB = C = [c_{ij}], \quad c_{ij} = \sum_{k=1}^n a_{ik} b_{kj}.
$$

Thus, the entry in the $i$\-th row and $j$\-th column of $AB$ is the dot product of the $i$\-th row of $A$ with the $j$\-th column of $B$.
因此， $AB$ 第 $i$ 行、第 $j$ 列的条目是 $A$ 第 $i$ 行与 $B$ 第 $j$ 列的点积。

Example 2.2.3. Let
例 2.2.3. 设

$$
A = \begin{bmatrix}1 & 2 \\0 & 3\end{bmatrix}, \quadB = \begin{bmatrix}4 & -1 \\2 & 5\end{bmatrix}.
$$

Then
然后

$$
AB = \begin{bmatrix}1\cdot4 + 2\cdot2 & 1\cdot(-1) + 2\cdot5 \\0\cdot4 + 3\cdot2 & 0\cdot(-1) + 3\cdot5\end{bmatrix} =\begin{bmatrix}8 & 9 \\6 & 15\end{bmatrix}.
$$

Notice that matrix multiplication is not commutative in general: $AB \neq BA$. Sometimes $BA$ may not even be defined if dimensions do not align.
请注意，矩阵乘法通常不满足交换律： $AB \neq BA$ 。如果维度不一致，有时甚至可能无法定义 $BA$ 。

### Geometric Meaning
几何意义

Matrix multiplication corresponds to the composition of linear transformations. If $A$ transforms vectors in $\mathbb{R}^n$ and $B$ transforms vectors in $\mathbb{R}^p$, then $AB$ represents applying $B$ first, then $A$. This makes matrices the algebraic language of transformations.
矩阵乘法对应于线性变换的复合。如果 $A$ 变换 $\mathbb{R}^n$ 中的向量， $B$ 变换 $\mathbb{R}^p$ 中的向量，那么 $AB$ 表示先应用 $B$ ，然后再应用 $A$ 。这使得矩阵成为变换的代数语言。

### Notation
符号

*   Matrix sum: $A+B$.
    矩阵和： $A+B$ 。
*   Scalar multiple: $cA$.
    标量倍数： $cA$ 。
*   Product: $AB$, defined only when the number of columns of $A$ equals the number of rows of $B$.
    乘积： $AB$ ，仅当 $A$ 的列数等于 $B$ 的行数时才定义。

### Why this matters
为什么这很重要

Matrix multiplication is the core mechanism of linear algebra: it encodes how transformations combine, how systems of equations are solved, and how data flows in modern algorithms. Addition and scalar multiplication make matrices into a vector space, while multiplication gives them an algebraic structure rich enough to model geometry, computation, and networks.
矩阵乘法是线性代数的核心机制：它编码了变换的组合方式、方程组的求解方式以及现代算法中数据流动的方式。加法和标量乘法将矩阵转化为向量空间，而乘法则赋予矩阵丰富的代数结构，使其能够对几何、计算和网络进行建模。

### Exercises 2.2
练习 2.2

1.  Compute $A+B$ for
    计算 $A+B$

$$
A = \begin{bmatrix} 2 & 3 \\-1 & 0 \end{bmatrix}, \quadB = \begin{bmatrix} 4 & -2 \\5 & 7 \end{bmatrix}.
$$

2.  Find 3A where
    查找 3A

$$
A = \begin{bmatrix} 1 & -4 \\2 & 6 \end{bmatrix}.
$$

3.  Multiply
    乘

$$
A = \begin{bmatrix} 1 & 0 & 2 \\-1 & 3 & 1 \end{bmatrix}, \quadB = \begin{bmatrix} 2 & 1 \\0 & -1 \\3 & 4 \end{bmatrix}.
$$

4.  Verify with an explicit example that $AB \neq BA$.
    通过明确的例子来验证 $AB \neq BA$ 。
5.  Prove that matrix multiplication is distributive: $A(B+C) = AB + AC$.
    证明矩阵乘法是分配的： $A(B+C) = AB + AC$ 。

## 2.3 Transpose and Inverse
2.3 转置和逆

Two special operations on matrices-the transpose and the inverse-give rise to deep algebraic and geometric properties. The transpose rearranges a matrix by flipping it across its main diagonal, while the inverse, when it exists, acts as the undo operation for matrix multiplication.
矩阵的两种特殊运算——转置和逆——引出了深刻的代数和几何性质。转置通过沿矩阵主对角线翻转来重新排列矩阵，而逆（如果存在）则充当矩阵乘法的撤消操作。

### The Transpose
转置

The transpose of an $m \times n$ matrix $A = [a_{ij}]$ is the $n \times m$ matrix $A^T = [a_{ji}]$, obtained by swapping rows and columns.
$m \times n$ 矩阵 $A = [a_{ij}]$ 的转置是通过交换行和列获得的 $n \times m$ 矩阵 $A^T = [a_{ji}]$ 。

Formally,
正式地，

$$
(A^T)\_{ij} = a\_{ji}.
$$

Example 2.3.1. If
例 2.3.1. 如果

$$
A = \begin{bmatrix}1 & 4 & -2 \\0 & 3 & 5\end{bmatrix},
$$

then
然后

$$
A^T = \begin{bmatrix}1 & 0 \\4 & 3 \\-2 & 5\end{bmatrix}.
$$

Properties of the Transpose.
转置的属性。

1.  $(A^T)^T = A$.
2.  $(A+B)^T = A^T + B^T$.
3.  $(cA)^T = cA^T$, for scalar $c$.
    $(cA)^T = cA^T$ ，对于标量 $c$ 。
4.  $(AB)^T = B^T A^T$.

The last rule is crucial: the order reverses.
最后一条规则至关重要：顺序反转。

### The Inverse
逆向

A square matrix $A \in \mathbb{R}^{n \times n}$ is said to be invertible (or nonsingular) if there exists another matrix $A^{-1}$ such that
如果存在另一个矩阵 $A^{-1}$ 满足以下条件，则称方阵 $A \in \mathbb{R}^{n \times n}$ 可逆（或非奇异）

$$
AA^{-1} = A^{-1}A = I_n,
$$

where $I_n$ is the $n \times n$ identity matrix. In this case, $A^{-1}$ is called the inverse of $A$.
其中𝐼 𝑛 I n ​ 是 $n \times n$ 单位矩阵。在这种情况下， $A^{-1}$ 被称为 $A$ 的逆。

Not every matrix is invertible. A necessary condition is that $\det(A) \neq 0$, a fact that will be developed in Chapter 6.
并非所有矩阵都是可逆的。必要条件是 $\det(A) \neq 0$ ，我们将在第 6 章中进一步阐述。

Example 2.3.2. Let
例 2.3.2. 设

$$
A = \begin{bmatrix}1 & 2 \\3 & 4\end{bmatrix}.
$$

Its determinant is $\det(A) = (1)(4) - (2)(3) = -2 \neq 0$. The inverse is
它的行列式是 $\det(A) = (1)(4) - (2)(3) = -2 \neq 0$ 。逆是

$$
A^{-1} = \frac{1}{\det(A)} \begin{bmatrix}4 & -2 \\-3 & 1\end{bmatrix} =\begin{bmatrix}-2 & 1 \\1.5 & -0.5\end{bmatrix}.
$$

Verification:
确认：

$$
AA^{-1} = \begin{bmatrix}1 & 2 \\3 & 4\end{bmatrix}\begin{bmatrix}-2 & 1 \\1.5 & -0.5\end{bmatrix} =\begin{bmatrix}1 & 0 \\0 & 1\end{bmatrix}.
$$

### Geometric Meaning
几何意义

*   The transpose corresponds to reflecting a linear transformation across the diagonal. For vectors, it switches between row and column forms.
    转置对应于沿对角线反映线性变换。对于向量，它在行和列形式之间切换。
*   The inverse, when it exists, corresponds to reversing a linear transformation. For example, if $A$ scales and rotates vectors, $A^{-1}$ rescales and rotates them back.
    如果存在逆变换，则它对应于线性变换的逆变换。例如，如果 $A$ 缩放并旋转了矢量，则 $A^{-1}$ 会将其重新缩放并旋转回去。

### Notation
符号

*   Transpose: $A^T$.
    转置： $A^T$ 。
*   Inverse: $A^{-1}$, defined only for invertible square matrices.
    逆： $A^{-1}$ ，仅为可逆方阵定义。
*   Identity: $I_n$, acts as the multiplicative identity.
    身份：𝐼 𝑛 I n ​ ，充当乘法恒等式。

### Why this matters
为什么这很重要

The transpose allows us to define symmetric and orthogonal matrices, central to geometry and numerical methods. The inverse underlies the solution of linear systems, encoding the idea of undoing a transformation. Together, these operations set the stage for determinants, eigenvalues, and orthogonalization.
转置使我们能够定义对称矩阵和正交矩阵，这是几何和数值方法的核心。逆矩阵是线性系统解的基础，它蕴含着撤销变换的思想。这些运算共同为行列式、特征值和正交化奠定了基础。

### Exercises 2.3
练习 2.3

1.  Compute the transpose of
    计算转置

$$
A = \begin{bmatrix} 2 & -1 & 3 \\ 0 & 4 & 5 \end{bmatrix}.
$$

2.  Verify that $(AB)^T = B^T A^T$ for
    验证 $(AB)^T = B^T A^T$

$$
A = \begin{bmatrix}1 & 2 \\0 & 1 \end{bmatrix}, \quadB = \begin{bmatrix}3 & 4 \\5 & 6 \end{bmatrix}.
$$

3.  Determine whether
    确定是否

$$
C = \begin{bmatrix}2 & 1 \\4 & 2 \end{bmatrix}
$$

is invertible. If so, find $C^{-1}$.
可逆。如果可逆，则求 $C^{-1}$ 。

4.  Find the inverse of
    求逆

$$
D = \begin{bmatrix}0 & 1 \\-1 & 0 \end{bmatrix},
$$

and explain its geometric action on vectors in the plane.
并解释其对平面向量的几何作用。

5.  Prove that if $A$ is invertible, then so is $A^T$, and $(A^T)^{-1} = (A^{-1})^T$.
    证明如果 $A$ 可逆，则 $A^T$ 和 $(A^T)^{-1} = (A^{-1})^T$ 也可逆。

## 2.4 Special Matrices
2.4 特殊矩阵

Certain matrices occur so frequently in theory and applications that they are given special names. Recognizing their properties allows us to simplify computations and understand the structure of linear transformations more clearly.
某些矩阵在理论和应用中出现频率很高，因此被赋予了特殊的名称。了解它们的性质可以简化计算，并更清楚地理解线性变换的结构。

### The Identity Matrix
身份矩阵

The identity matrix $I_n$ is the $n \times n$ matrix with ones on the diagonal and zeros elsewhere:
单位矩阵𝐼 𝑛 I n ​ 是 $n \times n$ 矩阵，对角线上为 1，其他位置为 0：

$$
I_n = \begin{bmatrix}1 & 0 & \cdots & 0 \\0 & 1 & \cdots & 0 \\\vdots & \vdots & \ddots & \vdots \\0 & 0 & \cdots & 1\end{bmatrix}.
$$

It acts as the multiplicative identity:
它充当乘法恒等式：

$$
AI_n = I_nA = A, \quad \text{for all } A \in \mathbb{R}^{n \times n}.
$$

Geometrically, $I_n$ represents the transformation that leaves every vector unchanged.
从几何学上讲，𝐼 𝑛 I n ​ 表示保持每个向量不变的变换。

### Diagonal Matrices
对角矩阵

A diagonal matrix has all off-diagonal entries zero:
对角矩阵的所有非对角元素均为零：

$$
D = \begin{bmatrix}d_{11} & 0 & \cdots & 0 \\0 & d_{22} & \cdots & 0 \\\vdots & \vdots & \ddots & \vdots \\0 & 0 & \cdots & d_{nn}\end{bmatrix}.
$$

Multiplication by a diagonal matrix scales each coordinate independently:
与对角矩阵相乘可独立缩放每个坐标：

$$
D\mathbf{x} = (d_{11}x_1, d_{22}x_2, \dots, d_{nn}x_n).
$$

Example 2.4.1. Let
例 2.4.1. 设

$$
D = \begin{bmatrix} 2 & 0 & 0 \\0 & 3 & 0 \\0 & 0 & -1 \end{bmatrix}, \quad\mathbf{x} = \begin{bmatrix}1 \\4 \\-2 \end{bmatrix}.
$$

Then
然后

$$
D\mathbf{x} = \begin{bmatrix}2 \\12 \\2 \end{bmatrix}.
$$

### Permutation Matrices
置换矩阵

A permutation matrix is obtained by permuting the rows of the identity matrix. Multiplying a vector by a permutation matrix reorders its coordinates.
置换矩阵是通过对单位矩阵的行进行置换而得到的。将向量乘以置换矩阵会重新排序其坐标。

Example 2.4.2. Let
例 2.4.2. 设

$$
P = \begin{bmatrix}0 & 1 & 0 \\1 & 0 & 0 \\0 & 0 & 1\end{bmatrix}.
$$

Then
然后

$$
P\begin{bmatrix}a \\b \\c \end{bmatrix} =\begin{bmatrix} b \\a \\c \end{bmatrix}.
$$

Thus, $P$ swaps the first two coordinates.
因此， $P$ 交换前两个坐标。

Permutation matrices are always invertible; their inverses are simply their transposes.
置换矩阵总是可逆的；它们的逆只是它们的转置。

### Symmetric and Skew-Symmetric Matrices
对称矩阵和斜对称矩阵

A matrix is symmetric if
如果矩阵是对称的

$$
A^T = A,
$$

and skew-symmetric if Symmetric matrices appear in quadratic forms and optimization, while skew-symmetric matrices describe rotations and cross products in geometry.
如果对称矩阵出现在二次形式和优化中，则为斜对称，而斜对称矩阵描述几何中的旋转和叉积。

### Orthogonal Matrices
正交矩阵

A square matrix $Q$ is orthogonal if
方阵 $Q$ 是正交的，如果

$$
Q^T Q = QQ^T = I.
$$

Equivalently, the rows (and columns) of $Q$ form an orthonormal set. Orthogonal matrices preserve lengths and angles; they represent rotations and reflections.
等价地， $Q$ 的行（和列）构成一个正交集。正交矩阵保留长度和角度；它们表示旋转和反射。

Example 2.4.3. The rotation matrix in the plane:
例2.4.3. 平面内的旋转矩阵:

$$
R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta \\\sin\theta & \cos\theta\end{bmatrix}
$$

is orthogonal, since
是正交的，因为

$$
R(\theta)^T R(\theta) = I_2.
$$

### Why this matters
为什么这很重要

Special matrices serve as the building blocks of linear algebra. Identity matrices define the neutral element, diagonal matrices simplify computations, permutation matrices reorder data, symmetric and orthogonal matrices describe fundamental geometric structures. Much of modern applied mathematics reduces complex problems to operations involving these simple forms.
特殊矩阵是线性代数的基石。单位矩阵定义中性元素，对角矩阵简化计算，置换矩阵重新排序数据，对称矩阵和正交矩阵描述基本几何结构。许多现代应用数学将复杂问题简化为涉及这些简单形式的运算。

### Exercises 2.4
练习 2.4

1.  Show that the product of two diagonal matrices is diagonal, and compute an example.
    证明两个对角矩阵的乘积是对角的，并计算一个例子。
2.  Find the permutation matrix that cycles $(a,b,c)$ into $(b,c,a)$.
    找到将 $(a,b,c)$ 循环到 $(b,c,a)$ 的置换矩阵。
3.  Prove that every permutation matrix is invertible and its inverse is its transpose.
    证明每个置换矩阵都是可逆的，并且它的逆是它的转置。
4.  Verify that
    验证

$$
Q = \begin{bmatrix}0 & 1 \\-1 & 0 \end{bmatrix}
$$

is orthogonal. What geometric transformation does it represent? 5. Determine whether
是正交的。它代表什么几何变换？5. 判断

$$
A = \begin{bmatrix}2 & 3 \\3 & 2 \end{bmatrix}, \quadB = \begin{bmatrix}0 & 5 \\-5 & 0 \end{bmatrix}
$$

are symmetric, skew-symmetric, or neither.
是对称的、斜对称的，或者都不是。

# Chapter 3. Systems of Linear Equations
第 3 章线性方程组

## 3.1 Linear Systems and Solutions
3.1 线性系统及其解

One of the central motivations for linear algebra is solving systems of linear equations. These systems arise naturally in science, engineering, and data analysis whenever multiple constraints interact. Matrices provide a compact language for expressing and solving them.
线性代数的核心动机之一是求解线性方程组。在科学、工程和数据分析领域，当多个约束相互作用时，这类方程组自然而然地出现。矩阵提供了一种简洁的语言来表达和求解它们。

### Linear Systems
线性系统

A linear system consists of equations where each unknown appears only to the first power and with no products between variables. A general system of $m$ equations in $n$ unknowns can be written as:
线性系统由方程组成，其中每个未知数仅出现一次方，并且之间没有乘积 变量。包含 $n$ 个未知数的 $m$ 个方程的一般系统可以写成：

$$
\begin{aligned}a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1, \\a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2, \\&\vdots \\a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m.\end{aligned}
$$

Here the coefficients $a_{ij}$ and constants $b_i$ are scalars, and the unknowns are $x_1, x_2, \dots, x_n$.
这里系数𝑎 𝑖 𝑗 a 伊奇 ​ 和常数𝑏 𝑖 b i ​ 是标量，未知数是𝑥 1 , 𝑥 2 , … , 𝑥 𝑛 x 1 ​ ，x 2 ​ ，…，x n ​ .

### Matrix Form
矩阵形式

The system can be expressed compactly as:
该系统可以简洁地表示为：

$$
A\mathbf{x} = \mathbf{b},
$$

where
在哪里

*   $A \in \mathbb{R}^{m \times n}$ is the coefficient matrix $[a_{ij}]$,
    $A \in \mathbb{R}^{m \times n}$ 是系数矩阵 $[a_{ij}]$ ，
*   $\mathbf{x} \in \mathbb{R}^n$ is the column vector of unknowns,
    $\mathbf{x} \in \mathbb{R}^n$ 是未知数的列向量，
*   $\mathbf{b} \in \mathbb{R}^m$ is the column vector of constants.
    $\mathbf{b} \in \mathbb{R}^m$ 是常数列向量。

This formulation turns the problem of solving equations into analyzing the action of a matrix.
这个公式将解方程的问题转化为分析矩阵的作用。

Example 3.1.1. The system
例 3.1.1. 系统

$$
\begin{cases}x + 2y = 5, \\3x - y = 4\end{cases}
$$

can be written as
可以写成

$$
\begin{bmatrix} 1 & 2 \\ 3 & -1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix}=\begin{bmatrix} 5 \\ 4 \end{bmatrix}.
$$

### Types of Solutions
解决方案类型

A linear system may have:
线性系统可能有：

1.  No solution (inconsistent): The equations conflict. Example:
    无解（不一致）：方程式相互矛盾。例如：

$$
\begin{cases}x + y = 1 \\x + y = 2\end{cases}
$$

This system has no solution.
这个系统没有解决方案。

2.  Exactly one solution (unique): The system’s equations intersect at a single point.
    只有一个解（唯一）：系统方程在一个点相交。
    Example: The following coefficient matrix:
    例如：以下系数矩阵：

$$
\begin{bmatrix}1 & 2 \\3 & -1\end{bmatrix}
$$

has a unique solution.
有一个独特的解决方案。

3.  Infinitely many solutions: The equations describe overlapping constraints (e.g., multiple equations representing the same line or plane).
    无数个解：方程描述重叠的约束（例如，表示同一条线或平面的多个方程）。

The nature of the solution depends on the rank of $A$ and its relation to the augmented matrix $(A|\mathbf{b})$, which we will study later.
解的性质取决于 $A$ 的秩及其与增广矩阵 $(A|\mathbf{b})$ 的关系，我们稍后会研究。

### Geometric Interpretation
几何解释

*   In $\mathbb{R}^2$, each linear equation represents a line. Solving a system means finding intersection points of lines.
    在 $\mathbb{R}^2$ 中，每个线性方程代表一条直线。求解方程组意味着找到直线的交点。
*   In $\mathbb{R}^3$, each equation represents a plane. A system may have no solution (parallel planes), one solution (a unique intersection point), or infinitely many (a line of intersection).
    在 $\mathbb{R}^3$ 中，每个方程代表一个平面。一个方程组可能没有解（平行平面），可能有一个解（唯一的交点），也可能有无数个解（一条交线）。
*   In higher dimensions, the picture generalizes: solutions form intersections of hyperplanes.
    在更高维度中，该图概括为：解决方案形成超平面的交点。

### Why this matters
为什么这很重要

Linear systems are the practical foundation of linear algebra. They appear in balancing chemical reactions, circuit analysis, least-squares regression, optimization, and computer graphics. Understanding how to represent and classify their solutions is the first step toward systematic solution methods like Gaussian elimination.
线性系统是线性代数的实践基础。它们出现在平衡化学反应、电路分析、最小二乘回归、优化和计算机图形学中。了解如何表示和分类它们的解是迈向高斯消元法等系统求解方法的第一步。

### Exercises 3.1
练习3.1

1.  Write the following system in matrix form:
    将以下系统写成矩阵形式：

$$
\begin{cases}2x + 3y - z = 7, \\x - y + 4z = 1, \\3x + 2y + z = 5\end{cases}
$$

2.  Determine whether the system
    确定系统是否

$$
\begin{cases}x + y = 1, \\2x + 2y = 2\end{cases}
$$

has no solution, one solution, or infinitely many solutions.
有无解、有一个解或有无数个解。

3.  Geometrically interpret the system
    几何解释系统

$$
\begin{cases}x + y = 3, \\x - y = 1\end{cases}
$$

in the plane.
在飞机上。

4.  Solve the system
    解决系统

$$
\begin{cases}2x + y = 1, \\x - y = 4\end{cases}
$$

and check your solution.
并检查您的解决方案。

5.  In $\mathbb{R}^3$, describe the solution set of
    在 $\mathbb{R}^3$ 中，描述

$$
\begin{cases}x + y + z = 0, \\2x + 2y + 2z = 0\end{cases}
$$

What geometric object does it represent?
它代表什么几何对象？

## 3.2 Gaussian Elimination
3.2 高斯消元法

To solve linear systems efficiently, we use Gaussian elimination: a systematic method of transforming a system into a simpler equivalent one whose solutions are easier to see. The method relies on elementary row operations that preserve the solution set.
为了高效地求解线性方程组，我们使用高斯消元法：这是一种将方程组转化为更简单、更易解的等效方程的系统方法。该方法依赖于保留解集的基本行运算。

### Elementary Row Operations
初等行运算

On an augmented matrix $(A|\mathbf{b})$, we are allowed three operations:
对于增广矩阵 $(A|\mathbf{b})$ ，我们可以进行三种运算：

1.  Row swapping: interchange two rows.
    换行：交换两行。
2.  Row scaling: multiply a row by a nonzero scalar.
    行缩放：将一行乘以非零标量。
3.  Row replacement: replace one row by itself plus a multiple of another row.
    行替换：用一行本身加上另一行的倍数来替换一行。

These operations correspond to re-expressing equations in different but equivalent forms.
这些运算对应于以不同但等效的形式重新表达方程。

### Row Echelon Form
行梯队形式

A matrix is in row echelon form (REF) if:
如果满足以下条件，则矩阵为行阶梯形矩阵（REF）：

1.  All nonzero rows are above any zero rows.
    所有非零行均位于任何零行之上。
2.  Each leading entry (the first nonzero number from the left in a row) is to the right of the leading entry in the row above.
    每个前导条目（一行中从左边开始的第一个非零数字）位于上一行前导条目的右侧。
3.  All entries below a leading entry are zero.
    前导条目下面的所有条目都为零。

Further, if each leading entry is 1 and is the only nonzero entry in its column, the matrix is in reduced row echelon form (RREF).
此外，如果每个前导项都是 1，并且是其列中唯一的非零项，则矩阵为简化行阶梯形式 (RREF)。

### Algorithm of Gaussian Elimination
高斯消元法

1.  Write the augmented matrix for the system.
    写出系统的增广矩阵。
2.  Use row operations to create zeros below each pivot (the leading entry in a row).
    使用行运算在每个枢轴（一行中的前导条目）下方创建零。
3.  Continue column by column until the matrix is in echelon form.
    继续逐列进行，直到矩阵呈阶梯形式。
4.  Solve by back substitution: starting from the last pivot equation and working upward.
    通过反向代入来求解：从最后一个枢轴方程开始向上求解。

If we continue to RREF, the solution can be read off directly.
如果我们继续 RREF，则可以直接读出解决方案。

### Example
例子

Example 3.2.1. Solve
例 3.2.1. 求解

$$
\begin{cases}x + 2y - z = 3, \\2x + y + z = 7, \\3x - y + 2z = 4.\end{cases}
$$

Step 1. Augmented matrix
步骤1.增广矩阵

$$
\left[\begin{array}{ccc|c}1 & 2 & -1 & 3 \\2 & 1 & 1 & 7 \\3 & -1 & 2 & 4\end{array}\right].
$$

Step 2. Eliminate below the first pivot
步骤 2. 消除第一个枢轴以下

Subtract 2 times row 1 from row 2, and 3 times row 1 from row 3:
从第 2 行减去第 1 行的 2 倍，从第 3 行减去第 1 行的 3 倍：

$$
\left[\begin{array}{ccc|c}1 & 2 & -1 & 3 \\0 & -3 & 3 & 1 \\0 & -7 & 5 & -5\end{array}\right].
$$

Step 3. Pivot in column 2
步骤 3. 在第 2 列中进行透视

Divide row 2 by -3:
将第 2 行除以 -3：

$$
\left[\begin{array}{ccc|c}1 & 2 & -1 & 3 \\0 & 1 & -1 & -\tfrac{1}{3} \\0 & -7 & 5 & -5\end{array}\right].
$$

Add 7 times row 2 to row 3:
将第 2 行的 7 倍加到第 3 行：

$$
\left[\begin{array}{ccc|c}1 & 2 & -1 & 3 \\0 & 1 & -1 & -\tfrac{1}{3} \\0 & 0 & -2 & -\tfrac{22}{3}\end{array}\right].
$$

Step 4. Pivot in column 3
步骤 4. 在第 3 列中进行透视

Divide row 3 by -2:
将第 3 行除以 -2：

$$
\left[\begin{array}{ccc|c}1 & 2 & -1 & 3 \\0 & 1 & -1 & -\tfrac{1}{3} \\0 & 0 & 1 & \tfrac{11}{3}\end{array}\right].
$$

Step 5. Back substitution
步骤 5. 回代

From the last row:
从最后一行开始：

$$
z = \tfrac{11}{3}.
$$

Second row:
第二行：

$$
y - z = -\tfrac{1}{3} \implies y = -\tfrac{1}{3} + \tfrac{11}{3} = \tfrac{10}{3}.
$$

First row:
第一行：

$$
x + 2y - z = 3 \implies x + 2\cdot\tfrac{10}{3} - \tfrac{11}{3} = 3.
$$

So
所以

$$
x + \tfrac{20}{3} - \tfrac{11}{3} = 3 \implies x + 3 = 3 \implies x = 0.
$$

Solution:
解决方案：

$$
(x,y,z) = \big(0, \tfrac{10}{3}, \tfrac{11}{3}\big).
$$

### Why this matters
为什么这很重要

Gaussian elimination is the foundation of computational linear algebra. It reduces complex systems to a form where solutions are visible, and it forms the basis for algorithms used in numerical analysis, scientific computing, and machine learning.
高斯消元法是计算线性代数的基础。它将复杂系统简化为可见解的形式，并构成数值分析、科学计算和机器学习中使用的算法的基础。

### Exercises 3.2
练习 3.2

1.  Solve by Gaussian elimination:
    通过高斯消元法求解：

$$
\begin{cases}x + y = 2, \\2x - y = 0.\end{cases}
$$

2.  Reduce the following augmented matrix to REF:
    将以下增广矩阵简化为 REF：

$$
\left[\begin{array}{ccc|c}1 & 1 & 1 & 6 \\2 & -1 & 3 & 14 \\1 & 4 & -2 & -2\end{array}\right].
$$

3.  Show that Gaussian elimination always produces either:
    证明高斯消元法总是产生以下结果：

*   a unique solution,
    一个独特的解决方案，
*   infinitely many solutions, or
    无穷多个解，或者
*   a contradiction (no solution).
    矛盾（无解）。

4.  Use Gaussian elimination to find all solutions of
    使用高斯消元法找到所有解

$$
\begin{cases}x + y + z = 0, \\2x + y + z = 1.\end{cases}
$$

5.  Explain why pivoting (choosing the largest available pivot element) is useful in numerical computation.
    解释为什么枢轴旋转（选择最大的可用枢轴元素）在数值计算中很有用。

## 3.3 Rank and Consistency
3.3 等级和一致性

Gaussian elimination not only provides solutions but also reveals the structure of a linear system. Two key ideas are the rank of a matrix and the consistency of a system. Rank measures the amount of independent information in the equations, while consistency determines whether the system has at least one solution.
高斯消元法不仅能提供解，还能揭示线性系统的结构。两个关键概念是矩阵的秩和系统的一致性。秩衡量方程中独立信息的数量，而一致性则决定系统是否至少有一个解。

### Rank of a Matrix
矩阵的秩

The rank of a matrix is the number of leading pivots in its row echelon form. Equivalently, it is the maximum number of linearly independent rows or columns.
矩阵的秩是其行阶梯形中前导主元的个数。换句话说，它是线性无关的行或列的最大数量。

Formally,
正式地，

$$
\text{rank}(A) = \dim(\text{row space of } A) = \dim(\text{column space of } A).
$$

The rank tells us the effective dimension of the space spanned by the rows (or columns).
秩告诉我们行（或列）所跨越的空间的有效维度。

Example 3.3.1. For
例 3.3.1. 对于

$$
A = \begin{bmatrix}1 & 2 & 3 \\2 & 4 & 6 \\3 & 6 & 9\end{bmatrix},
$$

row reduction gives
行减少给出

$$
\begin{bmatrix}1 & 2 & 3 \\0 & 0 & 0 \\0 & 0 & 0\end{bmatrix}.
$$

Thus, $\text{rank}(A) = 1$, since all rows are multiples of the first.
因此， $\text{rank}(A) = 1$ ，因为所有行都是第一行的倍数。

### Consistency of Linear Systems
线性系统的一致性

Consider the system $A\mathbf{x} = \mathbf{b}$. The system is consistent (has at least one solution) if and only if
考虑系统 $A\mathbf{x} = \mathbf{b}$ 。该系统是一致的（至少有一个解），当且仅当

$$
\text{rank}(A) = \text{rank}(A|\mathbf{b}),
$$

where $(A|\mathbf{b})$ is the augmented matrix. If the ranks differ, the system is inconsistent.
其中 $(A|\mathbf{b})$ 是增广矩阵。如果秩不同，则系统不一致。

*   If $\text{rank}(A) = \text{rank}(A|\mathbf{b}) = n$ (number of unknowns), the system has a unique solution.
    如果 $\text{rank}(A) = \text{rank}(A|\mathbf{b}) = n$ （未知数），则系统有一个唯一的解。
*   If $\text{rank}(A) = \text{rank}(A|\mathbf{b}) < n$, the system has infinitely many solutions.
    如果 $\text{rank}(A) = \text{rank}(A|\mathbf{b}) < n$ ，则系统有无数个解。

### Example
例子

Example 3.3.2. Consider
例 3.3.2. 考虑

$$
\begin{cases}x + y + z = 1, \\2x + 2y + 2z = 2, \\x + y + z = 3.\end{cases}
$$

The augmented matrix is
增广矩阵是

$$
\left[\begin{array}{ccc|c}1 & 1 & 1 & 1 \\2 & 2 & 2 & 2 \\1 & 1 & 1 & 3\end{array}\right].
$$

Row reduction gives
行减少给出

$$
\left[\begin{array}{ccc|c}1 & 1 & 1 & 1 \\0 & 0 & 0 & 0 \\0 & 0 & 0 & 2\end{array}\right].
$$

Here, $\text{rank}(A) = 1$, but $\text{rank}(A|\mathbf{b}) = 2$. Since the ranks differ, the system is inconsistent: no solution exists.
这里， $\text{rank}(A) = 1$ ，但 $\text{rank}(A|\mathbf{b}) = 2$ 。由于秩不同，系统不一致：不存在解。

### Example with Infinite Solutions
无限解的例子

Example 3.3.3. For
例 3.3.3. 对于

$$
\begin{cases}x + y = 2, \\2x + 2y = 4,\end{cases}
$$

the augmented matrix reduces to
增广矩阵简化为

$$
\left[\begin{array}{cc|c}1 & 1 & 2 \\0 & 0 & 0\end{array}\right].
$$

Here, $\text{rank}(A) = \text{rank}(A|\mathbf{b}) = 1 < 2$. Thus, infinitely many solutions exist, forming a line.
这里， $\text{rank}(A) = \text{rank}(A|\mathbf{b}) = 1 < 2$ 。因此，存在无数个解，形成一条线。

### Why this matters
为什么这很重要

Rank is a measure of independence: it tells us how many truly distinct equations or directions are present. Consistency explains when equations align versus when they contradict. These concepts connect linear systems to vector spaces and prepare for the ideas of dimension, basis, and the Rank–Nullity Theorem.
秩是独立性的度量：它告诉我们有多少个真正不同的方程或方向。一致性解释了方程何时一致，何时矛盾。这些概念将线性系统与向量空间联系起来，并为维度、基和秩零定理的概念做好准备。

### Exercises 3.3
练习 3.3

1.  Compute the rank of
    计算

$$
A = \begin{bmatrix}1 & 2 & 1 \\0 & 1 & -1 \\2 & 5 & -1\end{bmatrix}.
$$

2.  Determine whether the system
    确定系统

$$
\begin{cases}x + y + z = 1, \\2x + 3y + z = 2, \\3x + 5y + 2z = 3\end{cases}
$$

is consistent.
是一致的。

3.  Show that the rank of the identity matrix $I_n$ is $n$.
    证明单位矩阵𝐼的秩 𝑛 I n ​ 是 $n$ 。
    
4.  Give an example of a system in $\mathbb{R}^3$ with infinitely many solutions, and explain why it satisfies the rank condition.
    给出 $\mathbb{R}^3$ 中具有无穷多个解的系统的例子，并解释它为什么满足秩条件。
    
5.  Prove that for any matrix $A \in \mathbb{R}^{m \times n}$, $\text{rank}(A) \leq \min(m,n).$
    证明对于任意矩阵 $A \in \mathbb{R}^{m \times n}$ ， $\text{rank}(A) \leq \min(m,n).$
    

## 3.4 Homogeneous Systems
3.4 均质系统

A homogeneous system is a linear system in which all constant terms are zero:
齐次系统是所有常数项都为零的线性系统：

$$
A\mathbf{x} = \mathbf{0},
$$

where $A \in \mathbb{R}^{m \times n}$, and $\mathbf{0}$ is the zero vector in $\mathbb{R}^m$.
其中 $A \in \mathbb{R}^{m \times n}$ ，且 $\mathbf{0}$ 是 $\mathbb{R}^m$ 中的零向量。

### The Trivial Solution
简单的解决方案

Every homogeneous system has at least one solution:
每个同质系统至少有一个解：

$$
\mathbf{x} = \mathbf{0}.
$$

This is called the trivial solution. The interesting question is whether *nontrivial solutions* (nonzero vectors) exist.
这被称为平凡解。有趣的问题是是否存在*非平凡解* （非零向量）。

### Existence of Nontrivial Solutions
非平凡解的存在性

Nontrivial solutions exist precisely when the number of unknowns exceeds the rank of the coefficient matrix:
当未知数的数量超过系数矩阵的秩时，就会存在非平凡解：

$$
\text{rank}(A) < n.
$$

In this case, there are infinitely many solutions, forming a subspace of $\mathbb{R}^n$. The dimension of this solution space is
在这种情况下，有无穷多个解，形成一个 $\mathbb{R}^n$ 的子空间。这个解空间的维度是

$$
\dim(\text{null}(A)) = n - \text{rank}(A),
$$

where null(A) is the set of all solutions to $A\mathbf{x} = 0$. This set is called the null space or kernel of $A$.
其中 null(A) 是 $A\mathbf{x} = 0$ 所有解的集合。该集合称为 $A$ 的零空间或零核。

### Example
例子

Example 3.4.1. Consider
例 3.4.1. 考虑

$$
\begin{cases}x + y + z = 0, \\2x + y - z = 0.\end{cases}
$$

The augmented matrix is
增广矩阵是

$$
\left[\begin{array}{ccc|c}1 & 1 & 1 & 0 \\2 & 1 & -1 & 0\end{array}\right].
$$

Row reduction:
行减少：

$$
\left[\begin{array}{ccc|c}1 & 1 & 1 & 0 \\0 & -1 & -3 & 0\end{array}\right]\quad\to\quad\left[\begin{array}{ccc|c}1 & 1 & 1 & 0 \\0 & 1 & 3 & 0\end{array}\right].
$$

So the system is equivalent to:
因此该系统等同于：

$$
\begin{cases}x + y + z = 0, \\y + 3z = 0.\end{cases}
$$

From the second equation, $y = -3z$. Substituting into the first: $x - 3z + z = 0 \implies x = 2z.$
从第二个方程得出 $y = -3z$ 。代入第一个方程： $x - 3z + z = 0 \implies x = 2z.$

Thus solutions are:
因此解决方案是：

$$
(x,y,z) = z(2, -3, 1), \quad z \in \mathbb{R}.
$$

The null space is the line spanned by the vector $(2, -3, 1)$.
零空间是向量 $(2, -3, 1)$ 所跨越的线。

### Geometric Interpretation
几何解释

The solution set of a homogeneous system is always a subspace of $\mathbb{R}^n$.
同质系统的解集始终是 $\mathbb{R}^n$ 的子空间。

*   If $\text{rank}(A) = n$, the only solution is the zero vector.
    如果为 $\text{rank}(A) = n$ ，则唯一的解就是零向量。
*   If $\text{rank}(A) = n-1$, the solution set is a line through the origin.
    如果为 $\text{rank}(A) = n-1$ ，则解集是一条过原点的线。
*   If $\text{rank}(A) = n-2$, the solution set is a plane through the origin.
    如果为 $\text{rank}(A) = n-2$ ，则解集是通过原点的平面。

More generally, the null space has dimension $n - \text{rank}(A)$, known as the nullity.
更一般地，零空间的维度为 $n - \text{rank}(A)$ ，称为零度。

### Why this matters
为什么这很重要

Homogeneous systems are central to understanding vector spaces, subspaces, and dimension. They lead directly to the concepts of kernel, null space, and linear dependence. In applications, homogeneous systems appear in equilibrium problems, eigenvalue equations, and computer graphics transformations.
齐次系统是理解向量空间、子空间和维度的核心。它们直接引出核、零空间和线性相关性的概念。在实际应用中，齐次系统出现在平衡问题、特征值方程和计算机图形变换中。

### Exercises 3.4
练习 3.4

1.  Solve the homogeneous system
    解决均质系统

$$
\begin{cases}x + 2y - z = 0, \\2x + 4y - 2z = 0.\end{cases}
$$

What is the dimension of its solution space?
其解空间的维数是多少？

2.  Find all solutions of
    找到所有解决方案

$$
\begin{cases}x - y + z = 0, \\2x + y - z = 0.\end{cases}
$$

3.  Show that the solution set of any homogeneous system is a subspace of $\mathbb{R}^n$.
    证明任何同质系统的解集都是 $\mathbb{R}^n$ 的子空间。
    
4.  Suppose $A$ is a $3 \\times 3$matrix with$\\text{rank}(A) = 2$. What is the dimension of the null space of $A$?
    假设 $A$ 是 $3 \\times 3 $matrix with$ \\text{rank}(A) = 2 $. What is the dimension of the null space of $ A$？
    
5.  For
    为了
    

$$
A = \begin{bmatrix} 1 & 2 & -1 \\ 0 & 1 & 3 \end{bmatrix},
$$

compute a basis for the null space of $A$.
计算 $A$ 的零空间的基础。

# Chapter 4. Vector Spaces
第 4 章 向量空间

## 4.1 Definition of a Vector Space
4.1 向量空间的定义

Up to now we have studied vectors and matrices concretely in $\mathbb{R}^n$. The next step is to move beyond coordinates and define vector spaces in full generality. A vector space is an abstract setting where the familiar rules of addition and scalar multiplication hold, regardless of whether the elements are geometric vectors, polynomials, functions, or other objects.
到目前为止，我们已经在 $\mathbb{R}^n$ 中具体学习了向量和矩阵。下一步是超越坐标，全面定义向量空间。向量空间是一个抽象的场景，其中熟悉的加法和标量乘法规则始终成立，无论元素是几何向量、多项式、函数还是其他对象。

### Formal Definition
正式定义

A vector space over the real numbers $\mathbb{R}$ is a set $V$ equipped with two operations:
实数 $\mathbb{R}$ 上的向量空间是具有两个运算的集合 $V$ ：

1.  Vector addition: For any $\mathbf{u}, \mathbf{v} \in V$, there is a vector $\mathbf{u} + \mathbf{v} \in V$.
    向量加法：对于任何 $\mathbf{u}, \mathbf{v} \in V$ ，都有向量 $\mathbf{u} + \mathbf{v} \in V$ 。
2.  Scalar multiplication: For any scalar $c \in \mathbb{R}$ and any $\mathbf{v} \in V$, there is a vector $c\mathbf{v} \in V$.
    标量乘法：对于任何标量 $c \in \mathbb{R}$ 和任何 $\mathbf{v} \in V$ ，都有一个向量 $c\mathbf{v} \in V$ 。

These operations must satisfy the following axioms (for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and all scalars $a,b \in \mathbb{R}$):
这些运算必须满足以下公理（对于所有 $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ 和所有标量 $a,b \in \mathbb{R}$ ）：

1.  Commutativity of addition: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$.
    加法的交换性： $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ 。
2.  Associativity of addition: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$.
    加法的结合性： $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ 。
3.  Additive identity: There exists a zero vector $\mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$.
    加法恒等式：存在零向量 $\mathbf{0} \in V$ 使得 $\mathbf{v} + \mathbf{0} = \mathbf{v}$ 。
4.  Additive inverses: For each $\mathbf{v} \in V$, there exists $(-\mathbf{v} \in V$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$.
    加法逆元：对于每个 $\mathbf{v} \in V$ ，存在 $(-\mathbf{v} \in V$ 使得 $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ 。
5.  Compatibility of scalar multiplication: $a(b\mathbf{v}) = (ab)\mathbf{v}$.
    标量乘法的兼容性： $a(b\mathbf{v}) = (ab)\mathbf{v}$ 。
6.  Identity element of scalars: 1⋅v\=v.
    标量的标识元： 1⋅v\=v 。
7.  Distributivity over vector addition: $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$.
    向量加法的分配律： $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$ 。
8.  Distributivity over scalar addition: $(a+b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$.
    标量加法的分配律： $(a+b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$ 。

If a set $V$ with operations satisfies all eight axioms, we call it a vector space.
如果一个集合 $V$ 满足所有八个公理，我们称它为向量空间。

### Examples
示例

Example 4.1.1. Standard Euclidean space $\mathbb{R}^n$ with ordinary addition and scalar multiplication is a vector space. This is the model case from which the axioms are abstracted.
例 4.1.1. 标准欧几里得空间 $\mathbb{R}^n$ 进行普通的加法和标量乘法运算后，是一个向量空间。这是抽象出公理的典型例子。

Example 4.1.2. Polynomials The set of all polynomials with real coefficients, denoted $\mathbb{R}[x]$, forms a vector space. Addition and scalar multiplication are defined term by term.
例 4.1.2. 多项式 所有实系数多项式的集合，记为 $\mathbb{R}[x]$ ，构成一个向量空间。加法和标量乘法是逐项定义的。

Example 4.1.3. Functions The set of all real-valued functions on an interval, e.g. $f: [0,1] \to \mathbb{R}$, forms a vector space, since functions can be added and scaled pointwise.
例 4.1.3. 函数 区间上的所有实值函数的集合，例如 $f: [0,1] \to \mathbb{R}$ ，形成一个向量空间，因为函数可以逐点添加和缩放。

### Non-Examples
非示例

Not every set with operations qualifies. For instance, the set of positive real numbers under usual addition is not a vector space, because additive inverses (negative numbers) are missing. The axioms must all hold.
并非所有包含运算的集合都符合条件。例如，通常加法运算下的正实数集不是向量空间，因为缺少加法逆元（负数）。公理必须全部成立。

### Geometric Interpretation
几何解释

In familiar cases like $\mathbb{R}^2$ or $\mathbb{R}^3$, vector spaces provide the stage for geometry: vectors can be added, scaled, and combined to form lines, planes, and higher-dimensional structures. In abstract settings like function spaces, the same algebraic rules let us apply geometric intuition to infinite-dimensional problems.
在像 $\mathbb{R}^2$ 或 $\mathbb{R}^3$ 这样常见的情形下，向量空间为几何学提供了舞台：向量可以相加、缩放和组合，从而形成线、平面和更高维度的结构。在像函数空间这样的抽象环境中，同样的代数规则让我们能够将几何直觉应用于无限维问题。

### Why this matters
为什么这很重要

The concept of vector space unifies seemingly different mathematical objects under a single framework. Whether dealing with forces in physics, signals in engineering, or data in machine learning, the common language of vector spaces allows us to use the same techniques everywhere.
向量空间的概念将看似不同的数学对象统一在一个框架下。无论是处理物理学中的力、工程学中的信号，还是机器学习中的数据，向量空间的通用语言使我们能够在任何地方使用相同的技术。

### Exercises 4.1
练习4.1

1.  Verify that $\mathbb{R}^2$ with standard addition and scalar multiplication satisfies all eight vector space axioms.
    验证 $\mathbb{R}^2$ 通过标准加法和标量乘法满足所有八个向量空间公理。
2.  Show that the set of integers $\mathbb{Z}$ with ordinary operations is not a vector space over $\mathbb{R}$. Which axiom fails?
    证明：具有普通运算的整数集 $\mathbb{Z}$ 不是 $\mathbb{R}$ 上的向量空间。哪条公理不成立？
3.  Consider the set of all polynomials of degree at most 3. Show it forms a vector space over $\mathbb{R}$. What is its dimension?
    考虑所有次数最多为3的多项式的集合。证明它构成一个 $\mathbb{R}$ 上的向量空间。它的维度是多少？
4.  Give an example of a vector space where the vectors are not geometric objects.
    给出一个向量空间的例子，其中的向量不是几何对象。
5.  Prove that in any vector space, the zero vector is unique.
    证明在任何向量空间中，零向量都是唯一的。

## 4.2 Subspaces
4.2 子空间

A subspace is a smaller vector space living inside a larger one. Just as lines and planes naturally sit inside three-dimensional space, subspaces generalize these ideas to higher dimensions and more abstract settings.
子空间是位于较大向量空间中的较小向量空间。正如线和平面自然地存在于三维空间中一样，子空间将这些概念推广到更高维度和更抽象的场景。

### Definition
定义

Let $V$ be a vector space. A subset $W \subseteq V$ is called a subspace of $V$ if:
令 $V$ 为向量空间。若满足以下条件，则子集 $W \subseteq V$ 称为 $V$ 的子空间：

1.  $\mathbf{0} \in W$ (contains the zero vector),
    $\mathbf{0} \in W$ （包含零向量），
2.  For all $\mathbf{u}, \mathbf{v} \in W$, the sum $\mathbf{u} + \mathbf{v} \in W$ (closed under addition),
    对于所有 $\mathbf{u}, \mathbf{v} \in W$ ，总和为 $\mathbf{u} + \mathbf{v} \in W$ （加法闭包），
3.  For all scalars $c \in \mathbb{R}$ and vectors $\mathbf{v} \in W$, the product $c\mathbf{v} \in W$ (closed under scalar multiplication).
    对于所有标量 $c \in \mathbb{R}$ 和向量 $\mathbf{v} \in W$ ，乘积 $c\mathbf{v} \in W$ （在标量乘法下封闭）。

If these hold, then $W$ is itself a vector space with the inherited operations.
如果这些成立，那么 $W$ 本身就是具有继承操作的向量空间。

### Examples
示例

Example 4.2.1. Line through the origin in $\mathbb{R}^2$ The set
例 4.2.1. 穿过 $\mathbb{R}^2$ 中的原点的线 该套装

$$
W = \{ (t, 2t) \mid t \in \mathbb{R} \}
$$

is a subspace of $\mathbb{R}^2$. It contains the zero vector, is closed under addition, and is closed under scalar multiplication.
是 $\mathbb{R}^2$ 的一个子空间。它包含零向量，在加法运算下封闭，在标量乘法运算下封闭。

Example 4.2.2. The x–y plane in $\mathbb{R}^3$ The set
例 4.2.2. $\mathbb{R}^3$ 中的 x-y 平面 该套装

$$
W = \{ (x, y, 0) \mid x,y \in \mathbb{R} \}
$$

is a subspace of $\mathbb{R}^3$. It is the collection of all vectors lying in the plane through the origin parallel to the x–y plane.
是 $\mathbb{R}^3$ 的一个子空间。它是位于通过原点并平行于 x-y 平面的平面内的所有向量的集合。

Example 4.2.3. Null space of a matrix For a matrix $A \in \mathbb{R}^{m \times n}$, the null space
例 4.2.3. 矩阵的零空间 对于矩阵 $A \in \mathbb{R}^{m \times n}$ ，零空间

$$
\{ \mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0} \}
$$

is a subspace of $\mathbb{R}^n$. This subspace represents all solutions to the homogeneous system.
是 $\mathbb{R}^n$ 的一个子空间。该子空间表示齐次系统的所有解。

### Non-Examples
非示例

Not every subset is a subspace.
并非每个子集都是子空间。

*   The set ${ (x,y) \in \mathbb{R}^2 \mid x \geq 0 }$ is not a subspace: it is not closed under scalar multiplication (a negative scalar breaks the condition).
    集合 ${ (x,y) \in \mathbb{R}^2 \mid x \geq 0 }$ 不是子空间：它在标量乘法下不封闭（负标量会破坏该条件）。
*   Any line in $\mathbb{R}^2$ that does not pass through the origin is not a subspace, because it does not contain $\mathbf{0}$.
    $\mathbb{R}^2$ 中任何不经过原点的线都不是子空间，因为它不包含 $\mathbf{0}$ 。

### Geometric Interpretation
几何解释

Subspaces are the linear structures inside vector spaces.
子空间是向量空间内的线性结构。

*   In $\mathbb{R}^2$, the subspaces are: the zero vector, any line through the origin, or the entire plane.
    在 $\mathbb{R}^2$ 中，子空间是：零向量、过原点的任意直线或整个平面。
*   In $\mathbb{R}^3$, the subspaces are: the zero vector, any line through the origin, any plane through the origin, or the entire space.
    在 $\mathbb{R}^3$ 中，子空间是：零向量、过原点的任意直线、过原点的任意平面或整个空间。
*   In higher dimensions, the same principle applies: subspaces are the flat linear pieces through the origin.
    在更高的维度中，同样的原理适用：子空间是通过原点的平坦线性部分。

### Why this matters
为什么这很重要

Subspaces capture the essential structure of linear problems. Column spaces, row spaces, and null spaces are all subspaces. Much of linear algebra consists of understanding how these subspaces intersect, span, and complement each other.
子空间捕捉了线性问题的本质结构。列空间、行空间和零空间都是子空间。线性代数的大部分内容都在于理解这些子空间如何相互交叉、延伸和互补。

### Exercises 4.2
练习 4.2

1.  Prove that the set $W = { (x,0) \mid x \in \mathbb{R} } \subseteq \mathbb{R}^2$ is a subspace.
    证明集合 $W = { (x,0) \mid x \in \mathbb{R} } \subseteq \mathbb{R}^2$ 是一个子空间。
2.  Show that the line ${ (1+t, 2t) \mid t \in \mathbb{R} }$ is not a subspace of $\mathbb{R}^2$. Which condition fails?
    证明行 ${ (1+t, 2t) \mid t \in \mathbb{R} }$ 不是 $\mathbb{R}^2$ 的子空间。哪个条件不成立？
3.  Determine whether the set of all vectors $(x,y,z) \in \mathbb{R}^3$ satisfying $x+y+z=0$ is a subspace.
    确定满足 $x+y+z=0$ 的所有向量 $(x,y,z) \in \mathbb{R}^3$ 的集合是否为子空间。
4.  For the matrix
    对于矩阵

$$
A = \begin{bmatrix}1 & 2 & 3 \\4 & 5 & 6\end{bmatrix}
$$

Describe the null space of $A$ as a subspace of $\mathbb{R}^3$.
将 $A$ 的零空间描述为 $\mathbb{R}^3$ 的子空间。

5.  List all possible subspaces of $\mathbb{R}^2$.
    列出 $\mathbb{R}^2$ 所有可能的子空间。

## 4.3 Span, Basis, Dimension
4.3 跨度、基、维度

The ideas of span, basis, and dimension provide the language for describing the size and structure of subspaces. Together, they tell us how a vector space is generated, how many building blocks it requires, and how those blocks can be chosen.
跨度、基和维数的概念提供了描述子空间大小和结构的语言。它们共同告诉我们向量空间是如何生成的，它需要多少个构建块，以及如何选择这些构建块。

### Span
跨度

Given a set of vectors ${\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k} \subseteq V$, the span is the collection of all linear combinations:
给定一组向量 ${\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k} \subseteq V$ ，跨度是所有线性组合的集合：

$$
\text{span}\{\mathbf{v}_1, \dots, \mathbf{v}_k\} = \{ c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k \mid c_i \in \mathbb{R} \}.
$$

The span is always a subspace of $V$, namely the smallest subspace containing those vectors.
跨度始终是 $V$ 的子空间，即包含这些向量的最小子空间。

Example 4.3.1. In $\mathbb{R}^2$, $ \text{span}{(1,0)} = \{(x,0) \mid x \in \mathbb{R}\},$ the x-axis. Similarly, $\text{span}\{(1,0),(0,1)\} = \mathbb{R}^2.$
例 4.3.1。 在 $\mathbb{R}^2$ 中， $ \text{span}{(1,0)} = \{(x,0) \mid x \in \mathbb{R}\},$ x 轴。同样， $\text{span}\{(1,0),(0,1)\} = \mathbb{R}^2.$

### Basis
基础

A basis of a vector space $V$ is a set of vectors that:
向量空间 $V$ 的基是一组向量，其：

1.  Span $V$.
    跨度 $V$ 。
2.  Are linearly independent (no vector in the set is a linear combination of the others).
    是线性独立的（集合中没有向量是其他向量的线性组合）。

If either condition fails, the set is not a basis.
如果任一条件不成立，则该集合不作为基础。

Example 4.3.2. In $\mathbb{R}^3$, the standard unit vectors
例 4.3.2。 在 $\mathbb{R}^3$ 中，标准单位向量

$$
\mathbf{e}_1 = (1,0,0), \quad \mathbf{e}_2 = (0,1,0), \quad \mathbf{e}_3 = (0,0,1)
$$

form a basis. Every vector $(x,y,z)$ can be uniquely written as
构成基础。每个向量 $(x,y,z)$ 都可以唯一地写成

$$
x\mathbf{e}_1 + y\mathbf{e}_2 + z\mathbf{e}_3.
$$

### Dimension
方面

The dimension of a vector space $V$, written $\dim(V)$, is the number of vectors in any basis of $V$. This number is well-defined: all bases of a vector space have the same cardinality.
向量空间 $V$ 的维数，记作 $\dim(V)$ ，是任意 $V$ 的基中向量的数量。这个维数定义明确：向量空间的所有基都具有相同的基数。

Examples 4.3.3.
示例 4.3.3。

*   $\dim(\mathbb{R}^2) = 2$, with basis $(1,0), (0,1)$.
    $\dim(\mathbb{R}^2) = 2$ ，依据是 $(1,0), (0,1)$ 。
*   $\dim(\mathbb{R}^3) = 3$, with basis $(1,0,0), (0,1,0), (0,0,1)$.
    $\dim(\mathbb{R}^3) = 3$ ，依据是 $(1,0,0), (0,1,0), (0,0,1)$ 。
*   The set of polynomials of degree at most 3 has dimension 4, with basis $(1, x, x^2, x^3)$.
    次数最多为 3 的多项式集的维度为 4，基为 $(1, x, x^2, x^3)$ 。

### Geometric Interpretation
几何解释

*   The span is like the reach of a set of vectors.
    跨度就像一组向量的范围。
*   A basis is the minimal set of directions needed to reach everything in the space.
    基础是到达空间中所有事物所需的最小方向集。
*   The dimension is the count of those independent directions.
    维度是这些独立方向的数量。

Lines, planes, and higher-dimensional flats can all be described in terms of span, basis, and dimension.
线、平面和高维平面都可以用跨度、基和维度来描述。

### Why this matters
为什么这很重要

These concepts classify vector spaces and subspaces in terms of size and structure. Many theorems in linear algebra-such as the Rank–Nullity Theorem-are consequences of understanding span, basis, and dimension. In practical terms, bases are how we encode data in coordinates, and dimension tells us how much freedom a system truly has.
这些概念根据大小和结构对向量空间和子空间进行分类。线性代数中的许多定理，例如秩零定理，都是理解跨度、基和维数的结果。实际上，基是我们在坐标系中编码数据的方式，而维数则告诉我们一个系统真正拥有多少自由度。

### Exercises 4.3
练习 4.3

1.  Show that $(1,0,0)$, $(0,1,0)$, $(1,1,0)$ span the $xy$\-plane in $\mathbb{R}^3$. Are they a basis?
    证明 $(1,0,0)$ , $(0,1,0)$ , $(1,1,0)$ 在 $\mathbb{R}^3$ 中跨越 $xy$ -平面。它们是基吗？
2.  Find a basis for the line $\{(2t,-3t,t) : t \in \mathbb{R}\}$ in $\mathbb{R}^3$.
    找出 $\mathbb{R}^3$ 中第 $\{(2t,-3t,t) : t \in \mathbb{R}\}$ 行的依据。
3.  Determine the dimension of the subspace of $\mathbb{R}^3$ defined by $x+y+z=0$.
    确定由 $x+y+z=0$ 定义的 $\mathbb{R}^3$ 子空间的维数。
4.  Prove that any two different bases of $\mathbb{R}^n$ must contain exactly $n$ vectors.
    证明 $\mathbb{R}^n$ 的任意两个不同基必定包含恰好 $n$ 个向量。
5.  Give a basis for the set of polynomials of degree $\leq 2$. What is its dimension?
    给出次数为 $\leq 2$ 的多项式集的基。它的维数是多少？

## 4.4 Coordinates
4.4 坐标

Once a basis for a vector space is chosen, every vector can be expressed uniquely as a linear combination of the basis vectors. The coefficients in this combination are called the coordinates of the vector relative to that basis. Coordinates allow us to move between the abstract world of vector spaces and the concrete world of numbers.
一旦选定了向量空间的基，每个向量都可以唯一地表示为基向量的线性组合。该组合中的系数称为向量相对于该基的坐标。坐标使我们能够在向量空间的抽象世界和具体的数字世界之间移动。

### Coordinates Relative to a Basis
相对于基坐标

Let $V$ be a vector space, and let
令 $V$ 为向量空间，

$$
\mathcal{B} = \{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\}
$$

be an ordered basis for $V$. Every vector $\mathbf{u} \in V$ can be written uniquely as
是 $V$ 的有序基。每个向量 $\mathbf{u} \in V$ 都可以唯一地写成

$$
\mathbf{u} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_n \mathbf{v}_n.
$$

The scalars $(c_1, c_2, \dots, c_n)$ are the coordinates of $\mathbf{u}$ relative to $\mathcal{B}$, written
标量 $(c_1, c_2, \dots, c_n)$ 是 $\mathbf{u}$ 相对于 $\mathcal{B}$ 的坐标，写为

$$
[\mathbf{u}]_{\mathcal{B}} = \begin{bmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{bmatrix}.
$$

### Example in $\mathbb{R}^2$
$\mathbb{R}^2$ 中的示例

Example 4.4.1. Let the basis be
例 4.4.1. 设基础为

$$
\mathcal{B} = \{ (1,1), (1,-1) \}.
$$

To find the coordinates of $\mathbf{u} = (3,1)$ relative to $\mathcal{B}$, solve
要查找 $\mathbf{u} = (3,1)$ 相对于 $\mathcal{B}$ 的坐标，请求解

$$
(3,1) = c_1(1,1) + c_2(1,-1).
$$

This gives the system
这使得系统

$$
\begin{cases}c_1 + c_2 = 3, \\c_1 - c_2 = 1.\end{cases}
$$

Adding: $2c\_1 = 4 \\implies c\_1 = 2$. Then $c\_2 = 1$.
添加：$2c\_1 = 4 \\implies c\_1 = 2 $. Then $ c\_2 = 1$。

So,
所以，

$$
[\mathbf{u}]_{\mathcal{B}} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}.
$$

### Standard Coordinates
标准坐标

In $\mathbb{R}^n$, the standard basis is
在 $\mathbb{R}^n$ 中，标准依据是

$$
\mathbf{e}_1 = (1,0,\dots,0), \quad \mathbf{e}_2 = (0,1,0,\dots,0), \dots, \mathbf{e}_n = (0,\dots,0,1).
$$

Relative to this basis, the coordinates of a vector are simply its entries. Thus, column vectors are coordinate representations by default.
相对于此基，向量的坐标仅仅是它的元素。因此，列向量默认为坐标表示。

### Change of Basis
基础变更

If $\mathcal{B} = {\mathbf{v}_1, \dots, \mathbf{v}_n}$ is a basis of $\mathbb{R}^n$, the change of basis matrix is
如果𝐵 = 𝑣 1 , … , 𝑣 𝑛 B=v 1 ​ ，…，v n ​ 是 $\mathbb{R}^n$ 的基，基矩阵的变化是

$$
P = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n \end{bmatrix},
$$

with basis vectors as columns. For any vector $\mathbf{u}$,
以基向量为列。对于任意向量 $\mathbf{u}$ ，

$$
\mathbf{u} = P [\mathbf{u}]_{\mathcal{B}}, \qquad [\mathbf{u}]_{\mathcal{B}} = P^{-1}\mathbf{u}.
$$

Thus, switching between bases reduces to matrix multiplication.
因此，基数之间的切换就简化为矩阵乘法。

### Geometric Interpretation
几何解释

Coordinates are the address of a vector relative to a chosen set of directions. Different bases are like different coordinate systems: Cartesian, rotated, skewed, or scaled. The same vector may look very different numerically depending on the basis, but its geometric identity is unchanged.
坐标是向量相对于一组选定方向的地址。不同的基就像不同的坐标系：笛卡尔坐标系、旋转坐标系、倾斜坐标系或缩放坐标系。同一个向量在不同基上可能呈现出截然不同的数值，但其几何恒等式保持不变。

### Why this matters
为什么这很重要

Coordinates turn abstract vectors into concrete numerical data. Changing basis is the algebraic language for rotations of axes, diagonalization of matrices, and principal component analysis in data science. Mastery of coordinates is essential for moving fluidly between geometry, algebra, and computation.
坐标将抽象向量转化为具体的数值数据。变换基是数据科学中轴旋转、矩阵对角化和主成分分析的代数语言。掌握坐标系对于在几何、代数和计算之间流畅切换至关重要。

### Exercises 4.4
练习 4.4

1.  Express $(4,2)$ in terms of the basis $(1,1), (1,-1)$.
    根据基础 $(1,1), (1,-1)$ 表达 $(4,2)$ 。
2.  Find the coordinates of $(1,2,3)$ relative to the standard basis of $\mathbb{R}^3$.
    找出 $(1,2,3)$ 相对于 $\mathbb{R}^3$ 标准基的坐标。
3.  If $\mathcal{B} = \{(2,0), (0,3)\}$, compute $[ (4,6) ]_{\mathcal{B}}$.
    如果 $\mathcal{B} = \{(2,0), (0,3)\}$ ，则计算 \[ ( 4 , 6 ) \] 𝐵 \[(4,6)\] B ​ .
4.  Construct the change of basis matrix from the standard basis of $\mathbb{R}^2$ to $\mathcal{B} = \{(1,1), (1,-1)\}$.
    构建从标准基 $\mathbb{R}^2$ 到 $\mathcal{B} = \{(1,1), (1,-1)\}$ 的基变换矩阵。
5.  Prove that coordinate representation with respect to a basis is unique.
    证明关于基的坐标表示是唯一的。

# Chapter 5. Linear Transformations
第五章线性变换

## 5.1 Functions that Preserve Linearity
5.1 保持线性的函数

A central theme of linear algebra is understanding linear transformations: functions between vector spaces that preserve their algebraic structure. These transformations generalize the idea of matrix multiplication and capture the essence of linear behavior.
线性代数的核心主题是理解线性变换：向量空间之间保持其代数结构的函数。这些变换推广了矩阵乘法的概念，并抓住了线性行为的本质。

### Definition
定义

Let $V$ and $W$ be vector spaces over $\mathbb{R}$. A function
令 $V$ 和 $W$ 为 $\mathbb{R}$ 上的向量空间。函数

$$
T : V \to W
$$

is called a linear transformation (or linear map) if for all vectors $\mathbf{u}, \mathbf{v} \in V$ and all scalars $c \in \mathbb{R}$:
如果对于所有向量 $\mathbf{u}, \mathbf{v} \in V$ 和所有标量 $c \in \mathbb{R}$ ，则称为线性变换（或线性映射）：

1.  Additivity:
    加性：

$$
T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}),
$$

2.  Homogeneity:
    同质性：

$$
T(c\mathbf{u}) = cT(\mathbf{u}).
$$

If both conditions hold, then $T$ automatically respects linear combinations:
如果两个条件都成立，则 $T$ 自动遵循线性组合：

$$
T(c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k) = c_1 T(\mathbf{v}_1) + \cdots + c_k T(\mathbf{v}_k).
$$

### Examples
示例

Example 5.1.1. Scaling in $\mathbb{R}^2$. Let $T:\mathbb{R}^2 \to \mathbb{R}^2$ be defined by
例 5.1.1. 缩放 $\mathbb{R}^2$ 。令 $T:\mathbb{R}^2 \to \mathbb{R}^2$ 定义为

$$
T(x,y) = (2x, 2y).
$$

This doubles the length of every vector, preserving direction. It is linear.
这会使每个向量的长度加倍，同时保持方向不变。它是线性的。

Example 5.1.2. Rotation.
例 5.1.2. 旋转。

Let $R_\theta: \mathbb{R}^2 \to \mathbb{R}^2$ be
令 $R_\theta: \mathbb{R}^2 \to \mathbb{R}^2$ 为

$$
R_\theta(x,y) = (x\cos\theta - y\sin\theta, \; x\sin\theta + y\cos\theta).
$$

This rotates vectors by angle $\theta$. It satisfies additivity and homogeneity, hence is linear.
这将向量旋转角度 $\theta$ 。它满足可加性和齐次性，因此是线性的。

Example 5.1.3. Differentiation.
例 5.1.3. 区分。

Let $D: \mathbb{R}[x] \to \mathbb{R}[x]$ be differentiation: $D(p(x)) = p'(x)$.
令 $D: \mathbb{R}[x] \to \mathbb{R}[x]$ 为微分： $D(p(x)) = p'(x)$ 。

Since derivatives respect addition and scalar multiples, differentiation is a linear transformation.
由于导数尊重加法和标量倍数，因此微分是一种线性变换。

### Non-Example
非示例

The map $S:\mathbb{R}^2 \to \mathbb{R}^2$ defined by
地图 $S:\mathbb{R}^2 \to \mathbb{R}^2$ 定义为

$$
S(x,y) = (x^2, y^2)
$$

is not linear, because $S(\mathbf{u} + \mathbf{v}) \neq S(\mathbf{u}) + S(\mathbf{v})$ in general.
不是线性的，因为一般来说 $S(\mathbf{u} + \mathbf{v}) \neq S(\mathbf{u}) + S(\mathbf{v})$ 。

### Geometric Interpretation
几何解释

Linear transformations are exactly those that preserve the origin, lines through the origin, and proportions along those lines. They include familiar operations: scaling, rotations, reflections, shears, and projections. Nonlinear transformations bend or curve space, breaking these properties.
线性变换正是那些保留原点、过原点的直线以及沿这些直线的比例的变换。它们包括我们熟悉的操作：缩放、旋转、反射、剪切和投影。非线性变换会弯曲空间，从而破坏这些属性。

### Why this matters
为什么这很重要

Linear transformations unify geometry, algebra, and computation. They explain how matrices act on vectors, how data can be rotated or projected, and how systems evolve under linear rules. Much of linear algebra is devoted to understanding these transformations, their representations, and their invariants.
线性变换统一了几何、代数和计算。它解释了矩阵如何作用于向量，数据如何旋转或投影，以及系统如何在线性规则下演化。线性代数的大部分内容致力于理解这些变换、它们的表示及其不变量。

### Exercises 5.1
练习 5.1

1.  Verify that $T(x,y) = (3x-y, 2y)$ is a linear transformation on $\mathbb{R}^2$.
    验证 $T(x,y) = (3x-y, 2y)$ 是否是 $\mathbb{R}^2$ 的线性变换。
2.  Show that $T(x,y) = (x+1, y)$ is not linear. Which axiom fails?
    证明 $T(x,y) = (x+1, y)$ 不是线性的。哪条公理不成立？
3.  Prove that if $T$ and $S$ are linear transformations, then so is $T+S$.
    证明如果 $T$ 和 $S$ 是线性变换，那么 $T+S$ 也是线性变换。
4.  Give an example of a linear transformation from $\mathbb{R}^3$ to $\mathbb{R}^2$.
    给出一个从 $\mathbb{R}^3$ 到 $\mathbb{R}^2$ 的线性变换的例子。
5.  Let $T:\mathbb{R}[x] \to \mathbb{R}[x]$ be integration:
    令 $T:\mathbb{R}[x] \to \mathbb{R}[x]$ 为积分：

$$
T(p(x)) = \int_0^x p(t)\\,dt.
$$

Prove that $T$ is a linear transformation.
证明 $T$ 是线性变换。

## 5.2 Matrix Representation of Linear Maps
5.2 线性映射的矩阵表示

Every linear transformation between finite-dimensional vector spaces can be represented by a matrix. This correspondence is one of the central insights of linear algebra: it lets us use the tools of matrix arithmetic to study abstract transformations.
有限维向量空间之间的所有线性变换都可以用矩阵表示。这种对应关系是线性代数的核心洞见之一：它让我们能够利用矩阵运算工具来研究抽象的变换。

### From Linear Map to Matrix
从线性映射到矩阵

Let $T: \mathbb{R}^n \to \mathbb{R}^m$ be a linear transformation. Choose the standard basis $\{ \mathbf{e}_1, \dots, \mathbf{e}_n \}$ of $\mathbb{R}^n$, where $\mathbf{e}_i$ has a 1 in the $i$\-th position and 0 elsewhere.
令 $T: \mathbb{R}^n \to \mathbb{R}^m$ 为线性变换。选取 $\mathbb{R}^n$ 的标准基 $\{ \mathbf{e}_1, \dots, \mathbf{e}_n \}$ ，其中 𝑒 𝑖 e i ​ 第 $i$ 个位置为 1，其他地方为 0。

The action of $T$ on each basis vector determines the entire transformation:
$T$ 对每个基向量的作用决定了整个变换：

$$
T(\mathbf{e}\_j) = \begin{bmatrix}a_{1j} \\a_{2j} \\\vdots \\a_{mj} \end{bmatrix}.
$$

Placing these outputs as columns gives the matrix of $T$:
将这些输出作为列放置，得到矩阵 $T$ ：

$$
[T] = A = \begin{bmatrix}a_{11} & a_{12} & \cdots & a_{1n} \\a_{21} & a_{22} & \cdots & a_{2n} \\\vdots & \vdots & \ddots & \vdots \\a_{m1} & a_{m2} & \cdots & a_{mn}\end{bmatrix}.
$$

Then for any vector $\mathbf{x} \in \mathbb{R}^n$:
然后对于任意向量 $\mathbf{x} \in \mathbb{R}^n$ ：

$$
T(\mathbf{x}) = A\mathbf{x}.
$$

### Examples
示例

Example 5.2.1. Scaling in $\mathbb{R}^2$. Let $T(x,y) = (2x, 3y)$. Then
例 5.2.1. 缩放 $\mathbb{R}^2$ 。设 $T(x,y) = (2x, 3y)$ 。然后

$$
T(\mathbf{e}_1) = (2,0), \quad T(\mathbf{e}_2) = (0,3).
$$

So the matrix is
所以矩阵是

$$
[T] = \begin{bmatrix}2 & 0 \\0 & 3\end{bmatrix}.
$$

Example 5.2.2. Rotation in the plane. The rotation transformation $R_\theta(x,y) = (x\cos\theta - y\sin\theta, \; x\sin\theta + y\cos\theta)$ has matrix
例5.2.2. 平面旋转。 旋转变换 $R_\theta(x,y) = (x\cos\theta - y\sin\theta, \; x\sin\theta + y\cos\theta)$ 具有矩阵

$$
[R_\theta] = \begin{bmatrix}\cos\theta & -\sin\theta \\\sin\theta & \cos\theta\end{bmatrix}.
$$

Example 5.2.3. Projection onto the x-axis. The map $P(x,y) = (x,0)$ corresponds to
例 5.2.3. 投影到 x 轴。 地图 $P(x,y) = (x,0)$ 对应于

$$
[P] = \begin{bmatrix}1 & 0 \\0 & 0\end{bmatrix}.
$$

### Change of Basis
基础变更

Matrix representations depend on the chosen basis. If $\mathcal{B}$ and $\mathcal{C}$ are bases of $\mathbb{R}^n$ and $\mathbb{R}^m$, then the matrix of $T: \mathbb{R}^n \to \mathbb{R}^m$ with respect to these bases is obtained by expressing $T(\mathbf{v}_j)$ in terms of $\mathcal{C}$ for each $\mathbf{v}_j \in \mathcal{B}$. Changing bases corresponds to conjugating the matrix by the appropriate change-of-basis matrices.
矩阵表示取决于所选的基。如果 $\mathcal{B}$ 和 $\mathcal{C}$ 是 $\mathbb{R}^n$ 的基 和 $\mathbb{R}^m$ ，则 $T: \mathbb{R}^n \to \mathbb{R}^m$ 关于这些基的矩阵，可以通过将 $T(\mathbf{v}_j)$ 表示为 $\mathcal{C}$ 来获得，其中 $\mathbf{v}_j \in \mathcal{B}$ 表示为 $T(\mathbf{v}_j)$。改变基相当于将矩阵与适当的基变换矩阵共轭。

### Geometric Interpretation
几何解释

Matrices are not just convenient notation-they *are* linear maps once a basis is fixed. Every rotation, reflection, projection, shear, or scaling corresponds to multiplying by a specific matrix. Thus, studying linear transformations reduces to studying their matrices.
矩阵不仅仅是方便的符号——一旦基确定，它们*就是*线性映射。所有旋转、反射、投影、剪切或缩放都对应于乘以一个特定的矩阵。因此，研究线性变换可以归结为研究它们的矩阵。

### Why this matters
为什么这很重要

Matrix representations make linear transformations computable. They connect abstract definitions to explicit calculations, enabling algorithms for solving systems, finding eigenvalues, and performing decompositions. Applications from graphics to machine learning depend on this translation.
矩阵表示使线性变换可计算。它们将抽象定义与明确的计算联系起来，从而支持求解系统、查找特征值和执行分解的算法。从图形到机器学习等各种应用都依赖于这种转换。

### Exercises 5.2
练习 5.2

1.  Find the matrix representation of $T:\mathbb{R}^2 \to \mathbb{R}^2$, $T(x,y) = (x+y, x-y)$.
    找到 $T:\mathbb{R}^2 \to \mathbb{R}^2$ , $T(x,y) = (x+y, x-y)$ 的矩阵表示。
2.  Determine the matrix of the linear transformation $T:\mathbb{R}^3 \to \mathbb{R}^2$, $T(x,y,z) = (x+z, y-2z)$.
    确定线性变换矩阵 $T:\mathbb{R}^3 \to \mathbb{R}^2$ ， $T(x,y,z) = (x+z, y-2z)$ 。
3.  What matrix represents reflection across the line $y=x$ in $\mathbb{R}^2$?
    哪个矩阵表示 $\mathbb{R}^2$ 中沿线 $y=x$ 的反射？
4.  Show that the matrix of the identity transformation on $\mathbb{R}^n$ is $I_n$.
    证明 $\mathbb{R}^n$ 上的恒等变换矩阵是 𝐼 𝑛 I n ​ .
5.  For the differentiation map $D:\mathbb{R}_2[x] \to \mathbb{R}_1[x]$, where $\mathbb{R}_k[x]$ is the space of polynomials of degree at most $k$, find the matrix of $D$ relative to the bases $\{1,x,x^2\}$ and $\{1,x\}$.
    对于微分映射 $D:\mathbb{R}_2[x] \to \mathbb{R}_1[x]$ ，其中 $\mathbb{R}_k[x]$ 是次数最多为 $k$ 的多项式空间，求出 $D$ 相对于基数 $\{1,x,x^2\}$ 和 $\{1,x\}$ 的矩阵。

## 5.3 Kernel and Image
5.3 内核和镜像

To understand a linear transformation deeply, we must examine what it kills and what it produces. These ideas are captured by the kernel and the image, two fundamental subspaces associated with any linear map.
要深入理解线性变换，我们必须考察它消除了什么，又产生了什么。这些概念可以通过核和像来理解，它们是任何线性映射都相关的两个基本子空间。

### The Kernel
内核

The kernel (or null space) of a linear transformation $T: V \to W$ is the set of all vectors in $V$ that map to the zero vector in $W$:
线性变换 $T: V \to W$ 的核（或零空间）是 $V$ 中映射到 $W$ 中的零向量的所有向量的集合：

$$
\ker(T) = \{ \mathbf{v} \in V \mid T(\mathbf{v}) = \mathbf{0} \}.
$$

The kernel is always a subspace of $V$. It measures the degeneracy of the transformation-directions that collapse to nothing.
核始终是 $V$ 的子空间。它衡量的是坍缩为零的变换方向的退化程度。

Example 5.3.1. Let $T:\mathbb{R}^3 \to \mathbb{R}^2$ be defined by
例 5.3.1。 让 $T:\mathbb{R}^3 \to \mathbb{R}^2$ 定义为

$$
T(x,y,z) = (x+y, y+z).
$$

In matrix form,
以矩阵形式，

$$
[T] = \begin{bmatrix}1 & 1 & 0 \\0 & 1 & 1\end{bmatrix}.
$$

To find the kernel, solve
要找到内核，请解决

$$
\begin{bmatrix}1 & 1 & 0 \\0 & 1 & 1\end{bmatrix}\begin{bmatrix} x \\ y \\ z \end{bmatrix}= \begin{bmatrix} 0 \\ 0 \end{bmatrix}.
$$

This gives the equations $x + y = 0$, $y + z = 0$. Hence $x = -y, z = -y$. The kernel is
由此得到方程 $x + y = 0$ ， $y + z = 0$ 。因此 $x = -y, z = -y$ 。核函数为

$$
\ker(T) = \{ (-t, t, -t) \mid t \in \mathbb{R} \},
$$

a line in $\mathbb{R}^3$.
$\mathbb{R}^3$ 中的一行。

### The Image
图像

The image (or range) of a linear transformation $T: V \to W$ is the set of all outputs:
线性变换 $T: V \to W$ 的图像（或范围）是所有输出的集合：

$$
\text{im}(T) = \{ T(\mathbf{v}) \mid \mathbf{v} \in V \} \subseteq W.
$$

Equivalently, it is the span of the columns of the representing matrix. The image is always a subspace of $W$.
等效地，它是表示矩阵的列的跨度。图像始终是 $W$ 的子空间。

Example 5.3.2. For the same transformation as above,
例 5.3.2. 对于与上述相同的变换，

$$
[T] = \begin{bmatrix}1 & 1 & 0 \\0 & 1 & 1\end{bmatrix},
$$

the columns are $(1,0)$, $(1,1)$, and $(0,1)$. Since $(1,1) = (1,0) + (0,1)$, the image is
列为 $(1,0)$ 、 $(1,1)$ 和 $(0,1)$ 。由于 $(1,1) = (1,0) + (0,1)$ ，因此图像为

$$
\text{im}(T) = \text{span}\{ (1,0), (0,1) \} = \mathbb{R}^2.
$$

### Dimension Formula (Rank–Nullity Theorem)
维度公式（秩-零度定理）

For a linear transformation $T: V \to W$ with $V$ finite-dimensional,
对于线性变换 $T: V \to W$ 且 $V$ 为有限维，

$$
\dim(\ker(T)) + \dim(\text{im}(T)) = \dim(V).
$$

This fundamental result connects the lost directions (kernel) with the achieved directions (image).
这个基本结果将丢失的方向（内核）与实现的方向（图像）联系起来。

### Geometric Interpretation
几何解释

*   The kernel describes how the transformation flattens space (e.g., projecting a 3D object onto a plane).
    内核描述了变换如何使空间变平坦（例如，将 3D 对象投影到平面上）。
*   The image describes the target subspace reached by the transformation.
    该图像描述了变换所达到的目标子空间。
*   The rank–nullity theorem quantifies the tradeoff: the more dimensions collapse, the fewer remain in the image.
    秩零定理量化了这种权衡：维度崩溃得越多，图像中剩余的维度就越少。

### Why this matters
为什么这很重要

Kernel and image capture the essence of a linear map. They classify transformations, explain when systems have unique or infinite solutions, and form the backbone of important results like the Rank–Nullity Theorem, diagonalization, and spectral theory.
核和图像捕捉了线性映射的本质。它们对变换进行分类，解释系统何时具有唯一或无限解，并构成秩零定理、对角化和谱理论等重要结果的支柱。

### Exercises 5.3
练习 5.3

1.  Find the kernel and image of $T:\mathbb{R}^2 \to \mathbb{R}^2$, $T(x,y) = (x-y, x+y)$.
    查找 $T:\mathbb{R}^2 \to \mathbb{R}^2$ 、 $T(x,y) = (x-y, x+y)$ 的核和图像。
2.  Let
    让

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \end{bmatrix}
$$

Find bases for $\ker(A)$ and $\text{im}(A)$. 3. For the projection map $P(x,y,z) = (x,y,0)$, describe the kernel and image. 4. Prove that $\ker(T)$ and $\text{im}(T)$ are always subspaces. 5. Verify the Rank–Nullity Theorem for the transformation in Example 5.3.1.
找到 $\ker(A)$ 和 $\text{im}(A)$ 的基。3. 对于投影图 $P(x,y,z) = (x,y,0)$ ，描述其核和图像。4. 证明 $\ker(T)$ 和 $\text{im}(T)$ 始终是子空间。5. 验证示例5.3.1中变换的秩零性定理。

## 5.4 Change of Basis
5.4 基础变更

Linear transformations can look very different depending on the coordinate system we use. The process of rewriting vectors and transformations relative to a new basis is called a change of basis. This concept lies at the heart of diagonalization, orthogonalization, and many computational techniques.
根据我们使用的坐标系，线性变换看起来可能非常不同。相对于新的基重写向量和变换的过程称为基变换。这个概念是对角化、正交化以及许多计算技术的核心。

### Coordinate Change
坐标变换

Suppose $V$ is an $n$\-dimensional vector space, and let $\mathcal{B} = \{\mathbf{v}_1, \dots, \mathbf{v}_n\}$ be a basis. Every vector $\mathbf{x} \in V$ has a coordinate vector $[\mathbf{x}]_{\mathcal{B}} \in \mathbb{R}^n$.
假设 $V$ 是一个 $n$ 维向量空间，设 $\mathcal{B} = \{\mathbf{v}_1, \dots, \mathbf{v}_n\}$ 为基。每个向量 $\mathbf{x} \in V$ 都有一个坐标向量 $[\mathbf{x}]_{\mathcal{B}} \in \mathbb{R}^n$ 。

If $P$ is the change-of-basis matrix from $\mathcal{B}$ to the standard basis, then
如果 $P$ 是从 $\mathcal{B}$ 到标准基的基变换矩阵，则

$$
\mathbf{x} = P [\mathbf{x}]_{\mathcal{B}}.
$$

Equivalently,
等价地，

$$
[\mathbf{x}]_{\mathcal{B}} = P^{-1} \mathbf{x}.
$$

Here, $P$ has the basis vectors of $\mathcal{B}$ as its columns:
这里， $P$ 以 $\mathcal{B}$ 的基向量作为其列：

$$
P = \begin{bmatrix}\mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n\end{bmatrix}.
$$

### Transformation of Matrices
矩阵变换

Let $T: V \to V$ be a linear transformation. Suppose its matrix in the standard basis is $A$. In the basis $\mathcal{B}$, the representing matrix becomes
令 $T: V \to V$ 为线性变换。假设其在标准基中的矩阵为 $A$ 。在基 $\mathcal{B}$ 中，表示矩阵变为

$$
[T]_{\mathcal{B}} = P^{-1} A P.
$$

Thus, changing basis corresponds to a similarity transformation of the matrix.
因此，改变基础对应于矩阵的相似变换。

### Example
例子

Example 5.4.1. Let $T:\mathbb{R}^2 \to \mathbb{R}^2$ be given by
例 5.4.1。 令 $T:\mathbb{R}^2 \to \mathbb{R}^2$ 为

$$
T(x,y) = (3x + y, x + y).
$$

In the standard basis, its matrix is
在标准基础上，其矩阵为

$$
A = \begin{bmatrix}3 & 1 \\1 & 1\end{bmatrix}.
$$

Now consider the basis $\mathcal{B} = \{ (1,1), (1,-1) \}$. The change-of-basis matrix is
现在考虑基 $\mathcal{B} = \{ (1,1), (1,-1) \}$ 。基变换矩阵为

$$
P = \begin{bmatrix}1 & 1 \\1 & -1\end{bmatrix}.
$$

Then
然后

$$
[T]_{\mathcal{B}} = P^{-1} A P.
$$

Computing gives
计算得出

$$
[T]_{\mathcal{B}} =\begin{bmatrix}4 & 0 \\0 & 0\end{bmatrix}.
$$

In this new basis, the transformation is diagonal: one direction is scaled by 4, the other collapsed to 0.
在这个新的基础上，变换是对角的：一个方向缩放 4，另一个方向折叠为 0。

### Geometric Interpretation
几何解释

Change of basis is like rotating or skewing your coordinate grid. The underlying transformation does not change, but its description in numbers becomes simpler or more complicated depending on the basis. Finding a basis that simplifies a transformation (often a diagonal basis) is a key theme in linear algebra.
基变换就像旋转或倾斜坐标网格。底层变换本身不会改变，但其数值描述会根据基的变化而变得更简单或更复杂。寻找能够简化变换的基（通常是对角基）是线性代数的一个关键主题。

### Why this matters
为什么这很重要

Change of basis connects the abstract notion of similarity to practical computation. It is the tool that allows us to diagonalize matrices, compute eigenvalues, and simplify complex transformations. In applications, it corresponds to choosing a more natural coordinate system-whether in geometry, physics, or machine learning.
基变换将相似性的抽象概念与实际计算联系起来。它使我们能够对矩阵进行对角化、计算特征值并简化复杂的变换。在实际应用中，它相当于选择一个更自然的坐标系——无论是在几何、物理还是机器学习领域。

### Exercises 5.4
练习 5.4

1.  Let
    让

$$
A = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}
$$

Compute its representation in the basis $\{(1,0),(1,1)\}$. 2. Find the change-of-basis matrix from the standard basis of $\mathbb{R}^2$ to $\{(2,1),(1,1)\}$. 3. Prove that similar matrices (related by $P^{-1}AP$) represent the same linear transformation under different bases. 4. Diagonalize the matrix
计算其在基 $\{(1,0),(1,1)\}$ 中的表示。2. 求出从 $\mathbb{R}^2$ 到 $\{(2,1),(1,1)\}$ 的标准基变换矩阵。3. 证明相似的矩阵（由 $P^{-1}AP$ 关联）在不同基下表示相同的线性变换。4. 对角化矩阵

$$
A = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

in the basis $\{(1,1),(1,-1)\}$. 5. In $\mathbb{R}^3$, let $\mathcal{B} = \{(1,0,0),(1,1,0),(1,1,1)\}$. Construct the change-of-basis matrix $P$ and compute $P^{-1}$.
在基 $\{(1,1),(1,-1)\}$ 中。5. 在 $\mathbb{R}^3$ 中，令 $\mathcal{B} = \{(1,0,0),(1,1,0),(1,1,1)\}$ 。构造基变换矩阵 $P$ 并计算 $P^{-1}$ 。

# Chapter 6. Determinants
第六章 决定因素

## 6.1 Motivation and Geometric Meaning
6.1 动机和几何意义

Determinants are numerical values associated with square matrices. At first they may appear as a complicated formula, but their importance comes from what they measure: determinants encode scaling, orientation, and invertibility of linear transformations. They bridge algebra and geometry.
行列式是与方阵相关的数值。乍一看，它们可能看起来像一个复杂的公式，但它们的重要性在于它们所测量的内容：行列式编码了线性变换的缩放、方向和可逆性。它们连接了代数和几何。

### Determinants of 2×2 Matrices
2×2 矩阵的行列式

For a 2×2 matrix
对于 2×2 矩阵

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix},
$$

the determinant is defined as
行列式定义为

$$
\det(A) = ad - bc.
$$

Geometric meaning: If $A$ represents a linear transformation of the plane, then $|\det(A)|$ is the area scaling factor. For example, if $\det(A) = 2$, areas of shapes are doubled. If $\det(A) = 0$, the transformation collapses the plane to a line: all area is lost.
几何含义：如果 $A$ 表示平面的线性变换，则 $|\det(A)|$ 是面积缩放因子。例如，如果 $\det(A) = 2$ ，形状的面积将加倍。如果 $\det(A) = 0$ ，变换将平面折叠成一条线：所有面积都将丢失。

### Determinants of 3×3 Matrices
3×3 矩阵的行列式

For
为了

$$
A = \begin{bmatrix}a & b & c \\d & e & f \\g & h & i\end{bmatrix},
$$

the determinant can be computed as
行列式可以计算为

$$
\det(A) = a(ei - fh) - b(di - fg) + c(dh - eg).
$$

Geometric meaning: In $\mathbb{R}^3$, $|\det(A)|$ is the volume scaling factor. If $\det(A) < 0$, orientation is reversed (a handedness flip), such as turning a right-handed coordinate system into a left-handed one.
几何含义：在 $\mathbb{R}^3$ 中， $|\det(A)|$ 是体积缩放因子。如果为 $\det(A) < 0$ ，则方向反转（即手性翻转），例如将右手坐标系转换为左手坐标系。

### General Case
一般情况

For $A \in \mathbb{R}^{n \times n}$, the determinant is a scalar that measures how the linear transformation given by $A$ scales n-dimensional volume.
对于 $A \in \mathbb{R}^{n \times n}$ ，行列式是一个标量，它衡量 $A$ 给出的线性变换如何缩放 n 维体积。

*   If $\det(A) = 0$: the transformation squashes space into a lower dimension, so $A$ is not invertible.
    如果 $\det(A) = 0$ ：变换将空间压缩到较低维度，因此 $A$ 不可逆。
*   If $\det(A) > 0$: volume is scaled by $\det(A)$, orientation preserved.
    如果是 $\det(A) > 0$ ：体积按 $\det(A)$ 缩放，方向保持不变。
*   If $\det(A) < 0$: volume is scaled by $|\det(A)|$, orientation reversed.
    如果是 $\det(A) < 0$ ：体积按 $|\det(A)|$ 缩放，方向反转。

### Visual Examples
视觉示例

1.  Shear in $\mathbb{R}^2$: $A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$. Then $\det(A) = 1$. The transformation slants the unit square into a parallelogram but preserves area.
    $\mathbb{R}^2$ 处的剪切： $A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ 。然后是 $\det(A) = 1$ 。变换将单位正方形倾斜为平行四边形，但保留面积。
    
2.  Projection in $\mathbb{R}^2$: $A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$. Then $\det(A) = 0$. The unit square collapses into a line segment: area vanishes.
    $\mathbb{R}^2$ 中的投影： $A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$ 。然后 $\det(A) = 0$ 。单位正方形坍缩成一条线段：面积消失。
    
3.  Rotation in $\mathbb{R}^2$: $R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$. Then $\det(R_\theta) = 1$. Rotations preserve area and orientation.
    $\mathbb{R}^2$ 中的旋转： $R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ 。然后 $\det(R_\theta) = 1$ 。旋转保留面积和方向。
    

### Why this matters
为什么这很重要

The determinant is not just a formula-it is a measure of transformation. It tells us whether a matrix is invertible, how it distorts space, and whether it flips orientation. This geometric insight makes the determinant indispensable in analysis, geometry, and applied mathematics.
行列式不仅仅是一个公式，它还是一种变换的度量。它告诉我们一个矩阵是否可逆，它如何扭曲空间，以及它是否会翻转方向。这种几何学上的洞察力使得行列式在分析、几何和应用数学中不可或缺。

### Exercises 6.1
练习 6.1

1.  Compute the determinant of
    计算行列式

$$
\begin{bmatrix} 2 & 3 \\ 1 & 4 \end{bmatrix}
$$

What area scaling factor does it represent? 2. Find the determinant of the shear matrix
它代表什么面积比例因子？2. 求剪切矩阵的行列式

$$
\begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}
$$

What happens to the area of the unit square? 3. For the $3 \\times 3matrix \[100020003\] Compute the determinant. How does it scale volume in\\mathbb{R}^3$?4. Show that any rotation matrix in $\\mathbb{R}^2 has determinant \\1. 5. Give an example of a \\2 \\times 2$matrix with determinant$\-1$. What geometric action does it represent?
单位正方形的面积会发生什么变化？ 3. 对于 $3 \\times 3 矩阵 \[100020003\] 计算行列式。\\mathbb{R}^3 $?4. Show that any rotation matrix in $ \\mathbb{R}^2 的行列式为 \\ 1 ，它如何缩放体积 ？5. 举一个 \\ 2 \\times 2 $matrix with determinant$ -1$ 的例子 。它代表什么几何作用？

## 6.2 Properties of Determinants
6.2 行列式的性质

Beyond their geometric meaning, determinants satisfy a collection of algebraic rules that make them powerful tools in linear algebra. These properties allow us to compute efficiently, test invertibility, and understand how determinants behave under matrix operations.
除了几何意义之外，行列式还满足一系列代数规则，使其成为线性代数中强大的工具。这些性质使我们能够高效计算、测试可逆性，并理解行列式在矩​​阵运算下的行为。

### Basic Properties
基本属性

Let $A, B \in \mathbb{R}^{n \times n}$, and let $c \in \mathbb{R}$. Then:
令 $A, B \in \mathbb{R}^{n \times n}$ ，令 $c \in \mathbb{R}$ 。然后：

1.  Identity:
    身份：

$$
\det(I_n) = 1.
$$

2.  Triangular matrices: If $A$ is upper or lower triangular, then
    三角矩阵： 如果 $A$ 是上三角或下三角，则

$$
\det(A) = a_{11} a_{22} \cdots a_{nn}.
$$

3.  Row/column swap: Interchanging two rows (or columns) multiplies the determinant by $-1$.
    行/列交换： 交换两行（或列）将行列式乘以 $-1$ 。
    
4.  Row/column scaling: Multiplying a row (or column) by a scalar $c$ multiplies the determinant by $c$.
    行/列缩放： 将行（或列）乘以标量 $c$ 会将行列式乘以 $c$ 。
    
5.  Row/column addition: Adding a multiple of one row to another does not change the determinant.
    行/列加法：将一行的倍数添加到另一行不会改变行列式。
    
6.  Transpose:
    转置：
    

$$
\det(A^T) = \det(A).
$$

7.  Multiplicativity:
    乘法性：

$$
\det(AB) = \det(A)\det(B).
$$

8.  Invertibility: $A$ is invertible if and only if $\det(A) \neq 0$.
    可逆性： 当且仅当 $\det(A) \neq 0$ 时， $A$ 才是可逆的。

### Example Computations
计算示例

Example 6.2.1. For
例 6.2.1. 对于

$$
A = \begin{bmatrix}2 & 0 & 0 \\1 & 3 & 0 \\-1 & 4 & 5\end{bmatrix},
$$

$A$ is lower triangular, so
$A$ 是下三角，所以

$$
\det(A) = 2 \cdot 3 \cdot 5 = 30.
$$

Example 6.2.2. Let
例 6.2.2. 设

$$
B = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quadC = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}.
$$

Then
然后

$$
\det(B) = 1\cdot 4 - 2\cdot 3 = -2, \quad \det(C) = -1.
$$

Since $CB$ is obtained by swapping rows of $B$,
由于 $CB$ 是通过交换 $B$ 的行获得的，

$$
\det(CB) = -\det(B) = 2.
$$

This matches the multiplicativity rule: $\det(CB) = \det(C)\det(B) = (-1)(-2) = 2.$
这符合乘法规则： $\det(CB) = \det(C)\det(B) = (-1)(-2) = 2.$

### Geometric Insights
几何洞察

*   Row swaps: flipping orientation of space.
    行交换：翻转空间的方向。
*   Scaling a row: stretching space in one direction.
    缩放一行：朝一个方向拉伸空间。
*   Row replacement: sliding hyperplanes without altering volume.
    行替换：滑动超平面而不改变体积。
*   Multiplicativity: performing two transformations multiplies their scaling factors.
    乘法性：执行两个变换会将它们的比例因子相乘。

These properties make determinants both computationally manageable and geometrically interpretable.
这些性质使得行列式既易于计算管理，又易于几何解释。

### Why this matters
为什么这很重要

Determinant properties connect computation with geometry and theory. They explain why Gaussian elimination works, why invertibility is equivalent to nonzero determinant, and why determinants naturally arise in areas like volume computation, eigenvalue theory, and differential equations.
行列式的性质将计算与几何和理论联系起来。它们解释了高斯消元法为何有效，可逆性为何等价于非零行列式，以及行列式为何自然地出现在体积计算、特征值理论和微分方程等领域。

### Exercises 6.2
练习 6.2

1.  Compute the determinant of
    计算行列式

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 0 & 0 & 2 \end{bmatrix}.
$$

2.  Show that if two rows of a square matrix are identical, then its determinant is zero.
    证明如果方阵的两行相同，则其行列式为零。
    
3.  Verify $\det(A^T) = \det(A)$ for
    验证 $\det(A^T) = \det(A)$
    

$$
A = \begin{bmatrix} 2 & -1 \\ 3 & 4 \end{bmatrix}.
$$

4.  If $A$ is invertible, prove that
    如果 $A$ 可逆，则证明

$$
\det(A^{-1}) = \frac{1}{\det(A)}.
$$

5.  Suppose $A$ is a $3\\times 3$matrix with$\\det(A) = 5$. What is $\\det(2A)$?
    假设 $A$ 是 $3\\times 3 $matrix with$ \\det(A) = 5 $. What is $ \\det(2A)$？

## 6.3 Cofactor Expansion
6.3 辅因子展开

While determinants of small matrices can be computed directly from formulas, larger matrices require a systematic method. The cofactor expansion (also known as Laplace expansion) provides a recursive way to compute determinants by breaking them into smaller ones.
虽然小矩阵的行列式可以直接通过公式计算，但较大的矩阵则需要系统的方法。余因子展开式（也称为拉普拉斯展开式）通过将行列式分解为更小的矩阵，提供了一种递归计算行列式的方法。

### Minors and Cofactors
小式和辅因子

For an $n \times n$ matrix $A = [a_{ij}]$:
对于 $n \times n$ 矩阵 $A = [a_{ij}]$ ：

*   The minor $M_{ij}$ is the determinant of the $(n-1) \times (n-1)$ matrix obtained by deleting the $i$\-th row and $j$ -th column of $A$.
    小调𝑀 𝑖 𝑗 M 伊奇 ​ 是删除第 $i$ 行和 $j$ 后得到的 $(n-1) \times (n-1)$ 矩阵的行列式 $A$ 的第列。
*   The cofactor $C_{ij}$ is defined by
    辅因子𝐶 𝑖 𝑗 C 伊奇 ​ 定义为

$$
C_{ij} = (-1)^{i+j} M_{ij}.
$$

The sign factor $(-1)^{i+j}$ alternates in a checkerboard pattern:
符号因子 $(-1)^{i+j}$ 以棋盘格图案交替出现：

$$
\begin{bmatrix}+ & - & + & - & \cdots \\- & + & - & + & \cdots \\+ & - & + & - & \cdots \\\vdots & \vdots & \vdots & \vdots & \ddots\end{bmatrix}.
$$

### Cofactor Expansion Formula
辅因式展开公式

The determinant of $A$ can be computed by expanding along any row or any column:
$A$ 的行列式可以通过沿任意行或任意列展开来计算：

$$
\det(A) = \sum_{j=1}^n a_{ij} C_{ij} \quad \text{(expansion along row \(i\))},
$$

 

$$
\det(A) = \sum_{i=1}^n a_{ij} C_{ij} \quad \text{(expansion along column \(j\))}.
$$

### Example
例子

Example 6.3.1. Compute
例 6.3.1. 计算

$$
A = \begin{bmatrix}1 & 2 & 3 \\0 & 4 & 5 \\1 & 0 & 6\end{bmatrix}.
$$

Expand along the first row:
沿第一行展开：

$$
\det(A) = 1 \cdot C_{11} + 2 \cdot C_{12} + 3 \cdot C_{13}.
$$

*   For $C_{11}$:
    对于𝐶 11 C 11 ​ :

$$
M_{11} = \det \begin{bmatrix} 4 & 5 \\ 0 & 6 \end{bmatrix} = 24
$$

so $C_{11} = (+1)(24) = 24$.
所以 $C_{11} = (+1)(24) = 24$ 。

*   For $C_{12}$:
    对于𝐶 12 C 12 ​ :

$$
M_{12} = \det \begin{bmatrix} 0 & 5 \\ 1 & 6 \end{bmatrix} = 0 - 5 = -5
$$

so $C_{12} = (-1)(-5) = 5$.
所以 $C_{12} = (-1)(-5) = 5$ 。

*   For $C_{13}$:
    对于𝐶 13 C 13 ​ :

$$
M_{13} = \det \begin{bmatrix} 0 & 4 \\ 1 & 0 \end{bmatrix} = 0 - 4 = -4
$$

so $C_{13} = (+1)(-4) = -4$.
所以 $C_{13} = (+1)(-4) = -4$ 。

Thus,
因此，

$$
\det(A) = 1(24) + 2(5) + 3(-4) = 24 + 10 - 12 = 22.
$$

### Properties of Cofactor Expansion
辅因子展开的性质

1.  Expansion along any row or column yields the same result.
    沿任意行或列扩展都会产生相同的结果。
2.  The cofactor expansion provides a recursive definition of determinant: a determinant of size $n$ is expressed in terms of determinants of size $n-1$.
    余因子展开提供了行列式的递归定义：大小为 $n$ 的行列式可以用大小为 $n-1$ 的行列式来表示。
3.  Cofactors are fundamental in constructing the adjugate matrix, which gives a formula for inverses:
    余因子是构造伴随矩阵的基础，它给出了逆的公式：

$$
A^{-1} = \frac{1}{\det(A)} \, \text{adj}(A), \quad \text{where adj}(A) = [C_{ji}].
$$

### Geometric Interpretation
几何解释

Cofactor expansion breaks down the determinant into contributions from sub-volumes defined by fixing one row or column at a time. Each cofactor measures how that row/column influences the overall volume scaling.
余因子展开将行列式分解为由每次固定一行或一列定义的子体积的贡献。每个余因子衡量该行/列对整体体积缩放的影响。

### Why this matters
为什么这很重要

Cofactor expansion generalizes the small-matrix formulas and provides a conceptual definition of determinants. While not the most efficient way to compute determinants for large matrices, it is essential for theory, proofs, and connections to adjugates, Cramer’s rule, and classical geometry.
余因子展开式推广了小矩阵公式，并提供了行列式的概念定义。虽然它并非计算大矩阵行列式的最有效方法，但它对于理论、证明以及与伴随项、克莱姆规则和古典几何的联系至关重要。

### Exercises 6.3
练习 6.3

1.  Compute the determinant of
    计算行列式

$$
\begin{bmatrix}2 & 0 & 1 \\3 & -1 & 4 \\1 & 2 & 0\end{bmatrix}
$$

by cofactor expansion along the first column.
通过沿第一列的余因子展开。

2.  Verify that expanding along the second row of Example 6.3.1 gives the same determinant.
    验证沿示例 6.3.1 的第二行展开是否给出相同的行列式。
    
3.  Prove that expansion along any row gives the same value.
    证明沿任何行展开都会给出相同的值。
    
4.  Show that if a row of a matrix is zero, then its determinant is zero.
    证明如果矩阵的某一行是零，那么它的行列式也是零。
    
5.  Use cofactor expansion to prove that $\det(A) = \det(A^T)$.
    使用余因子展开来证明 $\det(A) = \det(A^T)$ 。
    

## 6.4 Applications (Volume, Invertibility Test)
6.4 应用（体积、可逆性测试）

Determinants are not merely algebraic curiosities; they have concrete geometric and computational uses. Two of the most important applications are measuring volumes and testing invertibility of matrices.
行列式不仅仅是代数上的奇闻；它们有着具体的几何和计算用途。其中最重要的两个应用是测量体积和检验矩阵的可逆性。

### Determinants as Volume Scalers
决定因素作为体积标量

Given vectors $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n \in \mathbb{R}^n$, arrange them as columns of a matrix:
给定向量 $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n \in \mathbb{R}^n$ ，将它们排列为矩阵的列：

$$
A = \begin{bmatrix}| & | & & | \\\mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n \\| & | & & |\end{bmatrix}.
$$

Then $|\det(A)|$ equals the volume of the parallelepiped spanned by these vectors.
那么 $|\det(A)|$ 等于这些向量所跨越的平行六面体的体积。

*   In $\mathbb{R}^2$, $|\det(A)|$ gives the area of the parallelogram spanned by $\mathbf{v}_1, \mathbf{v}_2$.
    在 $\mathbb{R}^2$ 中， $|\det(A)|$ 给出由 𝑣 构成的平行四边形的面积 1 , 𝑣 2 v 1 ​ ，v 2 ​ .
*   In $\mathbb{R}^3$, $|\det(A)|$ gives the volume of the parallelepiped spanned by $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$.
    在 $\mathbb{R}^3$ 中， $|\det(A)|$ 给出平行六面体的体积，跨度为 𝑣 1 , 𝑣 2 , 𝑣 3 v 1 ​ ，v 2 ​ ，v 3 ​ .
*   In higher dimensions, it generalizes to $n$\-dimensional volume (hypervolume).
    在更高维度中，它可以推广到 $n$ 维体积（超体积）。

Example 6.4.1. Let
例 6.4.1. 设

$$
\mathbf{v}_1 = (1,0,0), \quad \mathbf{v}_2 = (1,1,0), \quad \mathbf{v}_3 = (1,1,1).
$$

Then
然后

$$
A = \begin{bmatrix}1 & 1 & 1 \\0 & 1 & 1 \\0 & 0 & 1\end{bmatrix}, \quad \det(A) = 1.
$$

So the parallelepiped has volume 1, even though the vectors are not orthogonal.
因此，即使向量不正交，平行六面体的体积也是 1 。

### Invertibility Test
可逆性测试

A square matrix $A$ is invertible if and only if $\det(A) \neq 0$.
方阵 $A$ 可逆当且仅当 $\det(A) \neq 0$ 。

*   If $\det(A) = 0$: the transformation collapses space into a lower dimension (area/volume is zero). No inverse exists.
    如果 $\det(A) = 0$ ：变换将空间塌缩至较低维度（面积/体积为零）。不存在逆变换。
*   If $\det(A) \neq 0$: the transformation scales volume by $|\det(A)|$, and is reversible.
    如果 $\det(A) \neq 0$ ：变换将体积缩放 $|\det(A)|$ ，并且是可逆的。

Example 6.4.2. The matrix
例 6.4.2. 矩阵

$$
B = \begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix}
$$

has determinant $\det(B) = 2 \cdot 2 - 4 \cdot 1 = 0$. Thus, $B$ is not invertible. Geometrically, the two column vectors are collinear, spanning only a line in $\mathbb{R}^2$.
行列式为 $\det(B) = 2 \cdot 2 - 4 \cdot 1 = 0$ 。因此， $B$ 不可逆。几何上，这两个列向量共线，在 $\mathbb{R}^2$ 中仅延伸一条线。

### Cramer’s Rule
克莱默规则

Determinants also provide an explicit formula for solving systems of linear equations when the matrix is invertible. For $A\mathbf{x} = \mathbf{b}$ with $A \in \mathbb{R}^{n \times n}$:
当矩阵可逆时，行列式还提供了求解线性方程组的明确公式。 对于带有 $A \in \mathbb{R}^{n \times n}$ 的 $A\mathbf{x} = \mathbf{b}$ ：

$$
x_i = \frac{\det(A_i)}{\det(A)},
$$

where $A_i$ is obtained by replacing the $i$\-th column of $A$ with $\mathbf{b}$. While inefficient computationally, Cramer’s rule highlights the determinant’s role in solutions and uniqueness.
其中𝐴 𝑖 A i ​ 通过将 $A$ 的第 $i$ 列替换为 $\mathbf{b}$ 得到。克莱姆规则虽然计算效率低下，但它凸显了行列式在解和唯一性方面的作用。

### Orientation
方向

The sign of $\det(A)$ indicates whether a transformation preserves or reverses orientation. For example, a reflection in the plane has determinant $-1$, flipping handedness.
$\det(A)$ 的符号表示变换是保持方向还是反转方向。例如，平面上的反射具有行列式 $-1$ ，即翻转旋向性。

### Why this matters
为什么这很重要

Determinants condense key information: they measure scaling, test invertibility, and track orientation. These insights are indispensable in geometry (areas and volumes), analysis (Jacobian determinants in calculus), and computation ( solving systems and checking singularity).
行列式浓缩了关键信息：它们测量缩放比例、检验可逆性并追踪方向。这些洞见在几何学（面积和体积）、分析学（微积分中的雅可比行列式）和计算学（求解系统和检查奇点）中都不可或缺。

### Exercises 6.4
练习 6.4

1.  Compute the area of the parallelogram spanned by $(2,1)$ and $(1,3)$.
    计算 $(2,1)$ 和 $(1,3)$ 所构成的平行四边形的面积。
    
2.  Find the volume of the parallelepiped spanned by $(1,0,0), (1,1,0), (1,1,1)$.
    求出 $(1,0,0), (1,1,0), (1,1,1)$ 所跨度的平行六面体的体积。
    
3.  Determine whether the matrix
    确定矩阵
    

$$
\begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}
$$

is invertible. Justify using determinants. 4. Use Cramer’s rule to solve
是可逆的。用行列式证明。4. 使用克莱姆规则求解

$$
\begin{cases}x + y = 3, \\2x - y = 0.\end{cases}
$$

5.  Explain geometrically why a determinant of zero implies no inverse exists.
    从几何角度解释为什么行列式为零意味着不存在逆元。

# Chapter 7. Inner Product Spaces
第七章内积空间

## 7.1 Inner Products and Norms
7.1 内积和范数

To extend the geometric ideas of length, distance, and angle beyond $\mathbb{R}^2$ and $\mathbb{R}^3$, we introduce inner products. Inner products provide a way of measuring similarity between vectors, while norms derived from them measure length. These concepts are the foundation of geometry inside vector spaces.
为了将长度、距离和角度的几何概念扩展到 $\mathbb{R}^2$ 和 $\mathbb{R}^3$ 之外，我们引入了内积。内积提供了一种度量向量之间相似性的方法，而由内积导出的范数则用于度量长度。这些概念是向量空间几何的基础。

### Inner Product
内积

An inner product on a real vector space $V$ is a function
实向量空间 $V$ 上的内积是一个函数

$$
\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}
$$

that assigns to each pair of vectors $(\mathbf{u}, \mathbf{v})$ a real number, subject to the following properties:
为每对向量 $(\mathbf{u}, \mathbf{v})$ 分配一个实数，并遵循以下属性：

1.  Symmetry: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle.$
    对称： $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle.$
    
2.  Linearity in the first argument: $\langle a\mathbf{u} + b\mathbf{w}, \mathbf{v} \rangle = a \langle \mathbf{u}, \mathbf{v} \rangle + b \langle \mathbf{w}, \mathbf{v} \rangle.$
    第一个参数的线性： $\langle a\mathbf{u} + b\mathbf{w}, \mathbf{v} \rangle = a \langle \mathbf{u}, \mathbf{v} \rangle + b \langle \mathbf{w}, \mathbf{v} \rangle.$
    
3.  Positive-definiteness: $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$, and equality holds if and only if $\mathbf{v} = \mathbf{0}$.
    正定性： $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$ ，且仅当 $\mathbf{v} = \mathbf{0}$ 时等式成立。
    

The standard inner product on $\mathbb{R}^n$ is the dot product:
$\mathbb{R}^n$ 上的标准内积是点积：

$$
\langle \mathbf{u}, \mathbf{v} \rangle = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n.
$$

### Norms
规范

The norm of a vector is its length, defined in terms of the inner product:
向量的范数是其长度，根据内积定义：

$$
\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}.
$$

For the dot product in $\mathbb{R}^n$:
对于 $\mathbb{R}^n$ 中的点积：

$$
\|(x_1, x_2, \dots, x_n)\| = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}.
$$

### Angles Between Vectors
向量之间的角度

The inner product allows us to define the angle $\theta$ between two nonzero vectors $\mathbf{u}, \mathbf{v}$ by
通过内积，我们可以定义两个非零向量 $\mathbf{u}, \mathbf{v}$ 之间的角度 $\theta$ ，即

$$
\cos \theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \, \|\mathbf{v}\|}.
$$

Thus, two vectors are orthogonal if $\langle \mathbf{u}, \mathbf{v} \rangle = 0$.
因此，若 $\langle \mathbf{u}, \mathbf{v} \rangle = 0$ ，则两个向量正交。

### Examples
示例

Example 7.1.1. In $\mathbb{R}^2$, with $\mathbf{u} = (1,2)$, $\mathbf{v} = (3,4)$:
例 7.1.1。 在 $\mathbb{R}^2$ 中，使用 $\mathbf{u} = (1,2)$ 、 $\mathbf{v} = (3,4)$ ：

$$
\langle \mathbf{u}, \mathbf{v} \rangle = 1\cdot 3 + 2\cdot 4 = 11.
$$

 

$$
\|\mathbf{u}\| = \sqrt{1^2 + 2^2} = \sqrt{5}, \quad \|\mathbf{v}\| = \sqrt{3^2 + 4^2} = 5.
$$

So,
所以，

$$
\cos \theta = \frac{11}{\sqrt{5}\cdot 5}.
$$

Example 7.1.2. In the function space $C[0,1]$, the inner product
例 7.1.2。 在函数空间 $C[0,1]$ 中，内积

$$
\langle f, g \rangle = \int_0^1 f(x) g(x)\, dx
$$

defines a length
定义长度

$$
\|f\| = \sqrt{\int_0^1 f(x)^2 dx}.
$$

This generalizes geometry to infinite-dimensional spaces.
这将几何学推广到无限维空间。

### Geometric Interpretation
几何解释

*   Inner product: measures similarity between vectors.
    内积：测量向量之间的相似性。
*   Norm: length of a vector.
    范数：向量的长度。
*   Angle: measure of alignment between two directions.
    角度：两个方向之间的对齐度量。

These concepts unify algebraic operations with geometric intuition.
这些概念将代数运算与几何直觉统一起来。

### Why this matters
为什么这很重要

Inner products and norms allow us to extend geometry into abstract vector spaces. They form the basis of orthogonality, projections, Fourier series, least squares approximation, and many applications in physics and machine learning.
内积和范数使我们能够将几何扩展到抽象向量空间。它们构成了正交性、投影、傅里叶级数、最小二乘近似以及物理学和机器学习中许多应用的基础。

### Exercises 7.1
练习 7.1

1.  Compute $\langle (2,-1,3), (1,4,0) \rangle$. Then find the angle between them.
    计算 $\langle (2,-1,3), (1,4,0) \rangle$ 。然后求出它们之间的角度。
    
2.  Show that $\|(x,y)\| = \sqrt{x^2+y^2}$ satisfies the properties of a norm.
    证明∥ ( 𝑥 , 𝑦 ) ∥ = 𝑥 2 + 𝑦 2 ∥(x,y)∥= x 2 +y 2 ​ 满足范数的性质。
    
3.  In $\mathbb{R}^3$, verify that $(1,1,0)$ and $(1,-1,0)$ are orthogonal.
    在 $\mathbb{R}^3$ 中，验证 $(1,1,0)$ 和 $(1,-1,0)$ 是否正交。
    
4.  In $C[0,1]$, compute $\langle f,g \rangle$ for $f(x)=x$, $g(x)=1$.
    在 $C[0,1]$ 中，计算 $f(x)=x$ 、 $g(x)=1$ 的 $\langle f,g \rangle$ 。
    
5.  Prove the Cauchy–Schwarz inequality:
    证明柯西-施瓦茨不等式：
    
    $$
    |\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \, \|\mathbf{v}\|.
    $$
    

## 7.2 Orthogonal Projections
7.2 正交投影

One of the most useful applications of inner products is the notion of orthogonal projection. Projection allows us to approximate a vector by another lying in a subspace, minimizing error in the sense of distance. This idea underpins geometry, statistics, and numerical analysis.
内积最有用的应用之一是正交投影的概念。投影使我们能够用子空间中的另一个向量来近似一个向量，从而最小化距离方向上的误差。这一思想是几何、统计学和数值分析的基础。

### Projection onto a Line
投影到线上

Let $\mathbf{u} \in \mathbb{R}^n$ be a nonzero vector. The line spanned by $\mathbf{u}$ is
令 $\mathbf{u} \in \mathbb{R}^n$ 为非零向量。 $\mathbf{u}$ 所构成的线段为

$$
L = \{ c\mathbf{u} \mid c \in \mathbb{R} \}.
$$

Given a vector $\mathbf{v}$, the projection of $\mathbf{v}$ onto $\mathbf{u}$ is the vector in $L$ closest to $\mathbf{v}$. Geometrically, it is the shadow of $\mathbf{v}$ on the line.
给定向量 $\mathbf{v}$ ， $\mathbf{v}$ 在 $\mathbf{u}$ 上的投影是 $L$ 中距离 $\mathbf{v}$ 最近的向量。从几何学上讲，它是 $\mathbf{v}$ 在线上的阴影。

The formula is
公式是

$$
\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\langle \mathbf{v}, \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} \, \mathbf{u}.
$$

The error vector $\mathbf{v} - \text{proj}_{\mathbf{u}}(\mathbf{v})$ is orthogonal to $\mathbf{u}$.
误差向量 $\mathbf{v} - \text{proj}_{\mathbf{u}}(\mathbf{v})$ 与 $\mathbf{u}$ 正交。

### Example 7.2.1
例 7.2.1

Let $\mathbf{u} = (1,2)$, $\mathbf{v} = (3,1)$.
令 $\mathbf{u} = (1,2)$ ， $\mathbf{v} = (3,1)$ 。

$$
\langle \mathbf{v}, \mathbf{u} \rangle = 3\cdot 1 + 1\cdot 2 = 5, \quad\langle \mathbf{u}, \mathbf{u} \rangle = 1^2 + 2^2 = 5.
$$

So
所以

$$
\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{5}{5}(1,2) = (1,2).
$$

The error vector is $(3,1) - (1,2) = (2,-1)$, which is orthogonal to $(1,2)$.
误差向量为 $(3,1) - (1,2) = (2,-1)$ ，与 $(1,2)$ 正交。

### Projection onto a Subspace
投影到子空间

Suppose $W \subseteq \mathbb{R}^n$ is a subspace with orthonormal basis $\{ \mathbf{w}_1, \dots, \mathbf{w}_k \}$. The projection of a vector $\mathbf{v}$ onto $W$ is
假设 $W \subseteq \mathbb{R}^n$ 是一个具有正交基 $\{ \mathbf{w}_1, \dots, \mathbf{w}_k \}$ 的子空间。向量 $\mathbf{v}$ 在 $W$ 上的投影为

$$
\text{proj}_{W}(\mathbf{v}) = \langle \mathbf{v}, \mathbf{w}_1 \rangle \mathbf{w}_1 + \cdots + \langle \mathbf{v}, \mathbf{w}_k \rangle \mathbf{w}_k.
$$

This is the unique vector in $W$ closest to $\mathbf{v}$. The difference $\mathbf{v} - \text{proj}_{W}(\mathbf{v})$ is orthogonal to all of $W$.
这是 $W$ 中与 $\mathbf{v}$ 最接近的唯一向量。差值 $\mathbf{v} - \text{proj}_{W}(\mathbf{v})$ 与所有 $W$ 正交。

### Least Squares Approximation
最小二乘近似

Orthogonal projection explains the method of least squares. To solve an overdetermined system $A\mathbf{x} \approx \mathbf{b}$, we seek the $\mathbf{x}$ that makes $A\mathbf{x}$ the projection of $\mathbf{b}$ onto the column space of $A$. This gives the normal equations
正交投影解释了最小二乘法。为了解决超定问题 系统 $A\mathbf{x} \approx \mathbf{b}$ ，我们寻找 $\mathbf{x}$ ，使得 $A\mathbf{x}$ 成为 $\mathbf{b}$ 在 $A$ 的列空间上的投影。这给出了正则方程

$$
A^T A \mathbf{x} = A^T \mathbf{b}.
$$

Thus, least squares is just projection in disguise.
因此，最小二乘法只是伪装的投影。

### Geometric Interpretation
几何解释

*   Projection finds the closest point in a subspace to a given vector.
    投影找到子空间中距离给定向量最近的点。
*   It minimizes distance (error) in the sense of Euclidean norm.
    它按照欧几里得范数的意义最小化距离（误差）。
*   Orthogonality ensures the error vector points directly away from the subspace.
    正交性确保误差向量直接指向远离子空间的方向。

### Why this matters
为什么这很重要

Orthogonal projection is central in both pure and applied mathematics. It underlies the geometry of subspaces, the theory of Fourier series, regression in statistics, and approximation methods in numerical linear algebra. Whenever we fit data with a simpler model, projection is at work.
正交投影在纯数学和应用数学中都至关重要。它是子空间几何、傅里叶级数理论、统计学中的回归以及数值线性代数中的近似方法的基础。每当我们用更简单的模型拟合数据时，投影就会发挥作用。

### Exercises 7.2
练习 7.2

1.  Compute the projection of $(2,3)$ onto the vector $(1,1)$.
    计算 $(2,3)$ 到向量 $(1,1)$ 的投影。
2.  Show that $\mathbf{v} - \text{proj}_{\mathbf{u}}(\mathbf{v})$ is orthogonal to $\mathbf{u}$.
    证明 $\mathbf{v} - \text{proj}_{\mathbf{u}}(\mathbf{v})$ 与 $\mathbf{u}$ 正交。
3.  Let $W = \text{span}\{(1,0,0), (0,1,0)\} \subseteq \mathbb{R}^3$. Find the projection of $(1,2,3)$ onto $W$.
    令 $W = \text{span}\{(1,0,0), (0,1,0)\} \subseteq \mathbb{R}^3$ 。求 $(1,2,3)$ 到 $W$ 的投影。
4.  Explain why least squares fitting corresponds to projection onto the column space of $A$.
    解释为什么最小二乘拟合对应于 $A$ 的列空间上的投影。
5.  Prove that projection onto a subspace $W$ is unique: there is exactly one closest vector in $W$ to a given $\mathbf{v}$.
    证明投影到子空间 $W$ 是唯一的：在 $W$ 中，有且仅有一个与给定 $\mathbf{v}$ 最接近的向量。

## 7.3 Gram–Schmidt Process
7.3 格拉姆-施密特过程

The Gram–Schmidt process is a systematic way to turn any linearly independent set of vectors into an orthonormal basis. This is especially useful because orthonormal bases simplify computations: inner products become simple coordinate comparisons, and projections take clean forms.
格拉姆-施密特过程是一种将任意线性无关的向量集转化为正交基的系统方法。这种方法尤其有用，因为正交基可以简化计算：内积变成了简单的坐标比较，并且投影呈现出清晰的形式。

### The Idea
理念

Given a linearly independent set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\}$ in an inner product space, we want to construct an orthonormal set $\{\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_n\}$ that spans the same subspace.
给定内积空间中一组线性无关的向量 $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\}$ ，我们想要构建一个跨越同一子空间的正交集 $\{\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_n\}$ 。

We proceed step by step:
我们一步步来：

1.  Start with $\mathbf{v}_1$, normalize it to get $\mathbf{u}_1$.
    从𝑣开始 1 v 1 ​ ，将其标准化得到𝑢 1 u 1 ​ .
2.  Subtract from $\mathbf{v}_2$ its projection onto $\mathbf{u}_1$, leaving a vector orthogonal to $\mathbf{u}_1$. Normalize to get $\mathbf{u}_2$.
    从𝑣中减去 2 v 2 ​ 它在𝑢上的投影 1 u 1 ​ ，留下一个与𝑢正交的向量 1 u 1 ​ . 标准化得到𝑢 2 u 2 ​ .
3.  For each $\mathbf{v}_k$, subtract projections onto all previously constructed $\mathbf{u}_1, \dots, \mathbf{u}_{k-1}$, then normalize.
    对于每个𝑣 𝑘 v k ​ ，减去所有先前构建的𝑢上的投影 1 , … , 𝑢 𝑘 − 1 u 1 ​ ，…，你 k−1 ​ ，然后标准化。

### The Algorithm
算法

For $k = 1, 2, \dots, n$:
对于 $k = 1, 2, \dots, n$ ：

$$
\mathbf{w}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} \langle \mathbf{v}_k, \mathbf{u}_j \rangle \mathbf{u}_j,
$$

 

$$
\mathbf{u}_k = \frac{\mathbf{w}_k}{\|\mathbf{w}_k\|}.
$$

The result $\{\mathbf{u}_1, \dots, \mathbf{u}_n\}$ is an orthonormal basis of the span of the original vectors.
结果 $\{\mathbf{u}_1, \dots, \mathbf{u}_n\}$ 是原始向量跨度的正交基。

### Example 7.3.1
例 7.3.1

Take $\mathbf{v}_1 = (1,1,0), \ \mathbf{v}_2 = (1,0,1), \ \mathbf{v}_3 = (0,1,1)$ in $\mathbb{R}^3$.
在 $\mathbb{R}^3$ 中乘坐 $\mathbf{v}_1 = (1,1,0), \ \mathbf{v}_2 = (1,0,1), \ \mathbf{v}_3 = (0,1,1)$ 。

1.  Normalize $\mathbf{v}_1$:
    标准化𝑣 1 v 1 ​ :

$$
\mathbf{u}_1 = \frac{1}{\sqrt{2}}(1,1,0).
$$

2.  Subtract projection of $\mathbf{v}_2$ on $\mathbf{u}_1$:
    减去𝑣的投影 2 v 2 ​ 在𝑢 1 u 1 ​ :

$$
\mathbf{w}_2 = \mathbf{v}_2 - \langle \mathbf{v}_2,\mathbf{u}_1 \rangle \mathbf{u}_1.
$$

 

$$
\langle \mathbf{v}_2,\mathbf{u}_1 \rangle = \frac{1}{\sqrt{2}}(1\cdot 1 + 0\cdot 1 + 1\cdot 0) = \tfrac{1}{\sqrt{2}}.
$$

So
所以

$$
\mathbf{w}_2 = (1,0,1) - \tfrac{1}{\sqrt{2}}\cdot \tfrac{1}{\sqrt{2}}(1,1,0)= (1,0,1) - \tfrac{1}{2}(1,1,0)= \left(\tfrac{1}{2}, -\tfrac{1}{2}, 1\right).
$$

Normalize:
规范化：

$$
\mathbf{u}_2 = \frac{1}{\sqrt{\tfrac{1}{4}+\tfrac{1}{4}+1}} \left(\tfrac{1}{2}, -\tfrac{1}{2}, 1\right)= \frac{1}{\sqrt{\tfrac{3}{2}}}\left(\tfrac{1}{2}, -\tfrac{1}{2}, 1\right).
$$

3.  Subtract projections from $\mathbf{v}_3$:
    从𝑣中减去投影 3 v 3 ​ :

$$
\mathbf{w}_3 = \mathbf{v}_3 - \langle \mathbf{v}_3,\mathbf{u}_1 \rangle \mathbf{u}_1 - \langle \mathbf{v}_3,\mathbf{u}_2 \rangle \mathbf{u}_2.
$$

After computing, normalize to obtain $\mathbf{u}_3$.
计算后，归一化得到𝑢 3 u 3 ​ .

The result is an orthonormal basis of the span of $\{\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3\}$.
结果是 $\{\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3\}$ 跨度的正交基。

### Geometric Interpretation
几何解释

Gram–Schmidt is like straightening out a set of vectors: you start with the original directions and adjust each new vector to be perpendicular to all previous ones. Then you scale to unit length. The process ensures orthogonality while preserving the span.
格拉姆-施密特变换就像拉直一组向量：从原始方向开始，调整每个新向量使其与所有先前的向量垂直。然后缩放到单位长度。这个过程确保了正交性，同时保留了跨度。

### Why this matters
为什么这很重要

Orthonormal bases simplify inner products, projections, and computations in general. They make coordinate systems easier to work with and are crucial in numerical methods, QR decomposition, Fourier analysis, and statistics (orthogonal polynomials, principal component analysis).
正交基可以简化内积、投影和一般计算。它们使坐标系更易于使用，并且在数值方法、QR 分解、傅里叶分析和统计学（正交多项式、主成分分析）中至关重要。

### Exercises 7.3
练习 7.3

1.  Apply Gram–Schmidt to $(1,0), (1,1)$ in $\mathbb{R}^2$.
    对 $\mathbb{R}^2$ 中的 $(1,0), (1,1)$ 应用 Gram–Schmidt 公式。
2.  Orthogonalize $(1,1,1), (1,0,1)$ in $\mathbb{R}^3$.
    在 $\mathbb{R}^3$ 中对 $(1,1,1), (1,0,1)$ 进行正交化。
3.  Prove that each step of Gram–Schmidt yields a vector orthogonal to all previous ones.
    证明 Gram-Schmidt 的每一步都会产生一个与所有前面的向量正交的向量。
4.  Show that Gram–Schmidt preserves the span of the original vectors.
    证明 Gram–Schmidt 保留了原始向量的跨度。
5.  Explain how Gram–Schmidt leads to the QR decomposition of a matrix.
    解释 Gram-Schmidt 如何导致矩阵的 QR 分解。

## 7.4 Orthonormal Bases
7.4 正交基

An orthonormal basis is a basis of a vector space in which all vectors are both orthogonal to each other and have unit length. Such bases are the most convenient possible coordinate systems: computations involving inner products, projections, and norms become exceptionally simple.
正交基是向量空间中的一种基，其中所有向量彼此正交且具有单位长度。这样的基是最方便的坐标系：涉及内积、投影和范数的计算变得异常简单。

### Definition
定义

A set of vectors $\{\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_n\}$ in an inner product space $V$ is called an orthonormal basis if
内积空间 $V$ 中的一组向量 $\{\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_n\}$ 称为正交基，若

1.  $\langle \mathbf{u}_i, \mathbf{u}_j \rangle = 0$ whenever $i \neq j$ (orthogonality),
    $\langle \mathbf{u}_i, \mathbf{u}_j \rangle = 0$ 每当 $i \neq j$ （正交性）
2.  $\|\mathbf{u}_i\| = 1$ for all $i$ (normalization),
    对所有 $i$ 进行 $\|\mathbf{u}_i\| = 1$ （规范化），
3.  The set spans $V$.
    该集合跨越 $V$ 。

### Examples
示例

Example 7.4.1. In $\mathbb{R}^2$, the standard basis
例 7.4.1. 在 $\mathbb{R}^2$ 中，标准基础

$$
\mathbf{e}_1 = (1,0), \quad \mathbf{e}_2 = (0,1)
$$

is orthonormal under the dot product.
在点积下是正交的。

Example 7.4.2. In $\mathbb{R}^3$, the standard basis
例 7.4.2. 在 $\mathbb{R}^3$ 中，标准基础

$$
\mathbf{e}_1 = (1,0,0), \quad \mathbf{e}_2 = (0,1,0), \quad \mathbf{e}_3 = (0,0,1)
$$

is orthonormal.
是正交的。

Example 7.4.3. Fourier basis on functions:
例 7.4.3. 函数的傅里叶基：

$$
\{1, \cos x, \sin x, \cos 2x, \sin 2x, \dots\}
$$

is an orthogonal set in the space of square-integrable functions on $[-\pi,\pi]$ with inner product
是 $[-\pi,\pi]$ 上平方可积函数空间中的正交集，具有内积

$$
\langle f,g \rangle = \int_{-\pi}^{\pi} f(x) g(x)\, dx.
$$

After normalization, it becomes an orthonormal basis.
经过归一化之后，它就变成了正交基。

### Properties
特性

1.  Coordinate simplicity: If $\{\mathbf{u}_1,\dots,\mathbf{u}_n\}$ is an orthonormal basis of $V$, then any vector $\mathbf{v}\in V$ has coordinates
    坐标简单性：如果 $\{\mathbf{u}_1,\dots,\mathbf{u}_n\}$ 是 $V$ 的正交基，则任何向量 $\mathbf{v}\in V$ 都有坐标
    
    $$
    [\mathbf{v}] = \begin{bmatrix} \langle \mathbf{v}, \mathbf{u}_1 \rangle \\ \vdots \\ \langle \mathbf{v}, \mathbf{u}_n \rangle \end{bmatrix}.
    $$
    
    That is, coordinates are just inner products.
    也就是说，坐标只是内积。
    
2.  Parseval’s identity: For any $\mathbf{v} \in V$,
    帕塞瓦尔的身份： 对于任意的 $\mathbf{v} \in V$ ，
    
    $$
    \|\mathbf{v}\|^2 = \sum_{i=1}^n |\langle \mathbf{v}, \mathbf{u}_i \rangle|^2.
    $$
    
3.  Projections: The orthogonal projection onto the span of $\\{\mathbf{u}_1,\dots,\mathbf{u}_k\\}$ is
    预测： 𝑢 跨度上的正交投影 1 , … , 𝑢 𝑘 u 1 ​ ，…，你 k ​ 是
    
    $$
    \text{proj}(\mathbf{v}) = \sum_{i=1}^k \langle \mathbf{v}, \mathbf{u}_i \rangle \mathbf{u}_i.
    $$
    

### Constructing Orthonormal Bases
构造正交基

*   Start with any linearly independent set, then apply the Gram–Schmidt process to obtain an orthonormal set spanning the same subspace.
    从任意线性无关集开始，然后应用 Gram-Schmidt 过程来获取跨越相同子空间的正交集。
*   In practice, orthonormal bases are often chosen for numerical stability and simplicity of computation.
    在实践中，通常选择正交基来实现数值稳定性和计算简单性。

### Geometric Interpretation
几何解释

An orthonormal basis is like a perfectly aligned and equally scaled coordinate system. Distances and angles are computed directly using coordinates without correction factors. They are the ideal rulers of linear algebra.
正交基就像一个完美对齐且等比例缩放的坐标系。距离和角度直接使用坐标计算，无需校正因子。它们是线性代数的理想标尺。

### Why this matters
为什么这很重要

Orthonormal bases simplify every aspect of linear algebra: solving systems, computing projections, expanding functions, diagonalizing symmetric matrices, and working with Fourier series. In data science, principal component analysis produces orthonormal directions capturing maximum variance.
正交基简化了线性代数的各个方面：求解系统、计算投影、展开函数、对角化对称矩阵以及处理傅里叶级数。在数据科学中，主成分分析可以生成正交方向，从而捕捉最大方差。

### Exercises 7.4
练习 7.4

1.  Verify that $(1/\\sqrt{2})(1,1)$ and $(1/\\sqrt{2})(1,-1)$ form an orthonormal basis of $\mathbb{R}^2$.
    验证 $(1/\\sqrt{2})(1,1)$ 和 $(1/\\sqrt{2})(1,-1)$ 是否构成 $\mathbb{R}^2$ 的正交基。
2.  Express $(3,4)$ in terms of the orthonormal basis $\{(1/\\sqrt{2})(1,1), (1/\\sqrt{2})(1,-1)\}$.
    用正交基 $\{(1/\\sqrt{2})(1,1), (1/\\sqrt{2})(1,-1)\}$ 表示 $(3,4)$ 。
3.  Prove Parseval’s identity for $\\mathbb{R}^n$ with the dot product.
    使用点积证明 $\\mathbb{R}^n$ 的帕塞瓦尔恒等式。
4.  Find an orthonormal basis for the plane $x+y+z=0$ in $\\mathbb{R}^3$.
    在 $\\mathbb{R}^3$ 中找出平面 $x+y+z=0$ 的正交基。
5.  Explain why orthonormal bases are numerically more stable than arbitrary bases in computations.
    解释为什么正交基在计算中比任意基在数值上更稳定。

# Chapter 8. Eigenvalues and eigenvectors
第 8 章 特征值和特征向量

## 8.1 Definitions and Intuition
8.1 定义和直觉

The concepts of eigenvalues and eigenvectors reveal the most fundamental behavior of linear transformations. They identify the special directions in which a transformation acts by simple stretching or compressing, without rotation or distortion.
特征值和特征向量的概念揭示了线性变换最基本的行为。它们通过简单的拉伸或压缩（不进行旋转或变形）来识别变换所作用的特定方向。

### Definition
定义

Let $T: V \to V$ be a linear transformation on a vector space $V$. A nonzero vector $\mathbf{v} \in V$ is called an eigenvector of $T$ if
令 $T: V \to V$ 为向量空间 $V$ 上的线性变换。非零向量 $\mathbf{v} \in V$ 称为 $T$ 的特征向量，若

$$
T(\mathbf{v}) = \lambda \mathbf{v}
$$

for some scalar $\lambda \in \mathbb{R}$ (or $\mathbb{C}$). The scalar $\lambda$ is the eigenvalue corresponding to $\mathbf{v}$.
某个标量 $\lambda \in \mathbb{R}$ （或 $\mathbb{C}$ ）。标量 $\lambda$ 是对应于 $\mathbf{v}$ 的特征值。

Equivalently, if $A$ is the matrix of $T$, then eigenvalues and eigenvectors satisfy
等价地，如果 $A$ 是 $T$ 的矩阵，则特征值和特征向量满足

$$
A\mathbf{v} = \lambda \mathbf{v}.
$$

### Basic Examples
基本示例

Example 8.1.1. Let
例 8.1.1. 设

$$
A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}.
$$

Then
然后

$$
A(1,0)^T = 2(1,0)^T, \quad A(0,1)^T = 3(0,1)^T.
$$

So $(1,0)$ is an eigenvector with eigenvalue $2$, and $(0,1) is an eigenvector with eigenvalue \\3$.
因此 $(1,0)$ 是特征值为 $2 的特征向量， $, and $ (0,1) 是特征值为 \\ 3$ 的特征向量 。

Example 8.1.2. Rotation matrix in $\mathbb{R}^2$:
例 8.1.2。 $\mathbb{R}^2$ 中的旋转矩阵：

$$
R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}.
$$

If $\theta \neq 0, \pi$, $R_\theta$ has no real eigenvalues: every vector is rotated, not scaled. Over $\mathbb{C}$, however, it has eigenvalues $e^{i\theta}, e^{-i\theta}$.
如果 $\theta \neq 0, \pi$ ，𝑅 𝜃 R θ ​ 没有实数特征值：每个向量都经过旋转，而不是缩放。然而，在 $\mathbb{C}$ 上，它的特征值为 $e^{i\theta}, e^{-i\theta}$ 。

### Algebraic Formulation
代数公式

Eigenvalues arise from solving the characteristic equation:
特征值由求解特征方程得出：

$$
\det(A - \lambda I) = 0.
$$

This polynomial in $\lambda$ is the characteristic polynomial. Its roots are the eigenvalues.
$\lambda$ 中的这个多项式是特征多项式。它的根就是特征值。

### Geometric Intuition
几何直觉

*   Eigenvectors are directions that remain unchanged in orientation under a transformation; only their length is scaled.
    特征向量是在变换下方向保持不变的方向；只有它们的长度被缩放。
*   Eigenvalues tell us the scaling factor along those directions.
    特征值告诉我们沿这些方向的缩放因子。
*   If a matrix has many independent eigenvectors, it can often be simplified (diagonalized) by changing basis.
    如果矩阵具有许多独立的特征向量，则通常可以通过改变基来简化（对角化）。

### Applications in Geometry and Science
几何和科学中的应用

*   Stretching along principal axes of an ellipse (quadratic forms).
    沿椭圆的主轴拉伸（二次型）。
*   Stable directions of dynamical systems.
    动力系统的稳定方向。
*   Principal components in statistics and machine learning.
    统计学和机器学习中的主要成分。
*   Quantum mechanics, where observables correspond to operators with eigenvalues.
    量子力学，其中可观测量对应于具有特征值的算子。

### Why this matters
为什么这很重要

Eigenvalues and eigenvectors are a bridge between algebra and geometry. They provide a lens for understanding linear transformations in their simplest form. Nearly every application of linear algebra-differential equations, statistics, physics, computer science-relies on eigen-analysis.
特征值和特征向量是代数和几何之间的桥梁。它们为理解最简形式的线性变换提供了一个视角。几乎所有线性代数的应用——微分方程、统计学、物理学、计算机科学——都依赖于特征分析。

### Exercises 8.1
练习 8.1

1.  Find the eigenvalues and eigenvectors of $\begin{bmatrix} 4 & 0 \\ 0 & -1 \end{bmatrix}$.
    找到特征值和特征向量 $\begin{bmatrix} 4 & 0 \\ 0 & -1 \end{bmatrix}$ .
2.  Show that every scalar multiple of an eigenvector is again an eigenvector for the same eigenvalue.
    证明特征向量的每个标量倍数又是同一特征值的特征向量。
3.  Verify that the rotation matrix $R_\theta$ has no real eigenvalues unless $\theta = 0$ or $\pi$.
    验证旋转矩阵𝑅 𝜃 R θ ​ 除非 $\theta = 0$ 或 $\pi$ ，否则没有实数特征值。
4.  Compute the characteristic polynomial of $\begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$.
    计算特征多项式 $\begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$ .
5.  Explain geometrically what eigenvectors and eigenvalues represent for the shear matrix $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$.
    从几何角度解释特征向量和特征值对于剪切矩阵的意义 $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ .

## 8.2 Diagonalization
8.2 对角化

A central goal in linear algebra is to simplify the action of a matrix by choosing a good basis. Diagonalization is the process of rewriting a matrix so that it acts by simple scaling along independent directions. This makes computations such as powers, exponentials, and solving differential equations far easier.
线性代数的核心目标是通过选择合适的基来简化矩阵的运算。对角化是将矩阵重写，使其能够沿独立方向进行简单的缩放。这使得幂、指数和微分方程等计算变得更加容易。

### Definition
定义

A square matrix $A \in \mathbb{R}^{n \times n}$ is diagonalizable if there exists an invertible matrix $P$ such that
如果存在可逆矩阵 $P$ 并且满足以下条件，则方阵 $A \in \mathbb{R}^{n \times n}$ 可对角化

$$
P^{-1} A P = D,
$$

where $D$ is a diagonal matrix.
其中 $D$ 是一个对角矩阵。

The diagonal entries of $D$ are eigenvalues of $A$, and the columns of $P$ are the corresponding eigenvectors.
$D$ 的对角线项是 $A$ 的特征值， $P$ 的列是相应的特征向量。

### When is a Matrix Diagonalizable?
矩阵何时可对角化？

*   A matrix is diagonalizable if it has $n$ linearly independent eigenvectors.
    如果矩阵具有 $n$ 个线性无关的特征向量，则该矩阵可对角化。
*   Equivalently, the sum of the dimensions of its eigenspaces equals $n$.
    等效地，其特征空间的维数之和等于 $n$ 。
*   Symmetric matrices (over $\mathbb{R}$) are always diagonalizable, with an orthonormal basis of eigenvectors.
    对称矩阵（在 $\mathbb{R}$ 上）始终可对角化，且具有特征向量的正交基。

### Example 8.2.1
例 8.2.1

Let
让

$$
A = \begin{bmatrix} 4 & 1 \\ 0 & 2 \end{bmatrix}.
$$

1.  Characteristic polynomial:
    特征多项式：

$$
\det(A - \lambda I) = (4-\lambda)(2-\lambda).
$$

So eigenvalues are $\lambda_1 = 4$, $\lambda_2 = 2$.
所以特征值是 $\lambda_1 = 4$ ， $\lambda_2 = 2$ 。

2.  Eigenvectors:
    特征向量：

*   For $\lambda = 4$, solve $(A-4I)\mathbf{v}=0$: $\begin{bmatrix} 0 & 1 \\ 0 & -2 \end{bmatrix}\mathbf{v} = 0$, giving $\mathbf{v}_1 = (1,0)$.
    对于 $\lambda = 4$ ，求解 $(A-4I)\mathbf{v}=0$ ： $\begin{bmatrix} 0 & 1 \\ 0 & -2 \end{bmatrix}\mathbf{v} = 0$ ，得到 $\mathbf{v}_1 = (1,0)$ 。
*   For $\lambda = 2$: $(A-2I)\mathbf{v}=0$, giving $\mathbf{v}_2 = (1,-2)$.
    对于 $\lambda = 2$ ： $(A-2I)\mathbf{v}=0$ ，给出 $\mathbf{v}_2 = (1,-2)$ 。

3.  Construct $P = \begin{bmatrix} 1 & 1 \\ 0 & -2 \end{bmatrix}$. Then
    构造 $P = \begin{bmatrix} 1 & 1 \\ 0 & -2 \end{bmatrix}$ 。然后

$$
P^{-1} A P = \begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix}.
$$

Thus, $A$ is diagonalizable.
因此， $A$ 是可对角化的。

### Why Diagonalize?
为什么要对角化？

*   Computing powers: If $A = P D P^{-1}$, then
    计算能力： 如果 $A = P D P^{-1}$ ，则
    
    $$
    A^k = P D^k P^{-1}.
    $$
    
    Since $D$ is diagonal, $D^k$ is easy to compute.
    由于 $D$ 是对角线，因此 $D^k$ 很容易计算。
    
*   Matrix exponentials: $e^A = P e^D P^{-1}$, useful in solving differential equations.
    矩阵指数： $e^A = P e^D P^{-1}$ ，有助于解决微分方程。
    
*   Understanding geometry: Diagonalization reveals the directions along which a transformation stretches or compresses space independently.
    理解几何：对角化揭示了变换独立拉伸或压缩空间的方向。
    

### Non-Diagonalizable Example
不可对角化的例子

Not all matrices can be diagonalized.
并非所有矩阵都可以对角化。

$$
A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
$$

has only one eigenvalue $\lambda = 1$, with eigenspace dimension 1. Since $n=2$ but we only have 1 independent eigenvector, $A$ is not diagonalizable.
只有一个特征值 $\lambda = 1$ ，特征空间维数为 1。由于 $n=2$ 但我们只有 1 个独立特征向量，因此 $A$ 不可对角化。

### Geometric Interpretation
几何解释

Diagonalization means we have found a basis of eigenvectors. In this basis, the matrix acts by simple scaling along each coordinate axis. It transforms complicated motion into independent 1D motions.
对角化意味着我们找到了特征向量的基。在此基上，矩阵通过沿每个坐标轴进行简单的缩放来发挥作用。它将复杂的运动转化为独立的一维运动。

### Why this matters
为什么这很重要

Diagonalization is a cornerstone of linear algebra. It simplifies computation, reveals structure, and is the starting point for the spectral theorem, Jordan form, and many applications in physics, engineering, and data science.
对角化是线性代数的基石。它简化了计算，揭示了结构，并且是谱定理、若尔当形式以及物理、工程和数据科学中许多应用的起点。

### Exercises 8.2
练习 8.2

1.  Diagonalize
    对角化
    
    $$
    A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}.
    $$
    
2.  Determine whether
    确定是否
    
    $$
    A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
    $$
    
    is diagonalizable. Why or why not?
    是可对角化的。为什么或为什么不？
    
3.  Find $A^5$ for
    查找 $A^5$
    
    $$
    A = \begin{bmatrix} 4 & 1 \\ 0 & 2 \end{bmatrix}
    $$
    
    using diagonalization.
    使用对角化。
    
4.  Show that any $n \times n$ matrix with $n$ distinct eigenvalues is diagonalizable.
    证明任何具有 $n$ 个不同特征值的 $n \times n$ 矩阵都是可对角化的。
    
5.  Explain why real symmetric matrices are always diagonalizable.
    解释为什么实对称矩阵总是可对角化的。
    

## 8.3 Characteristic Polynomials
8.3 特征多项式

The key to finding eigenvalues is the characteristic polynomial of a matrix. This polynomial encodes the values of $\lambda$ for which the matrix $A - \lambda I$ fails to be invertible.
寻找特征值的关键是矩阵的特征多项式。该多项式对值进行编码 矩阵 $A - \lambda I$ 不可逆，其中 $\lambda$ 。

### Definition
定义

For an $n \times n$ matrix $A$, the characteristic polynomial is
对于 $n \times n$ 矩阵 $A$ ，特征多项式为

$$
p_A(\lambda) = \det(A - \lambda I).
$$

The roots of $p_A(\lambda)$ are the eigenvalues of $A$.
$p_A(\lambda)$ 的根是 $A$ 的特征值。

### Examples
示例

Example 8.3.1. Let
例 8.3.1. 设

$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}.
$$

Then
然后

$$
p_A(\lambda) = \det\!\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix}= (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3.
$$

Thus eigenvalues are $\lambda = 1, 3$.
因此特征值为 $\lambda = 1, 3$ 。

Example 8.3.2. For
例 8.3.2. 对于

$$
A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

(rotation by 90°),
（旋转 90°），

$$
p_A(\lambda) = \det\!\begin{bmatrix} -\lambda & -1 \\ 1 & -\lambda \end{bmatrix}= \lambda^2 + 1.
$$

Eigenvalues are $\lambda = \pm i$. No real eigenvalues exist, consistent with pure rotation.
特征值为 $\lambda = \pm i$ 。不存在实数特征值，与纯旋转一致。

Example 8.3.3. For a triangular matrix
例 8.3.3. 对于三角矩阵

$$
A = \begin{bmatrix} 2 & 1 & 0 \\ 0 & 3 & 5 \\ 0 & 0 & 4 \end{bmatrix},
$$

the determinant is simply the product of diagonal entries minus $\lambda$:
行列式仅仅是对角线项的乘积减去 $\lambda$ ：

$$
p_A(\lambda) = (2-\lambda)(3-\lambda)(4-\lambda).
$$

So eigenvalues are 2,3,4.
所以特征值为 2,3,4 。

### Properties
特性

1.  The characteristic polynomial of an $n \times n$ matrix has degree $n$.
    $n \times n$ 矩阵的特征多项式的度为 $n$ 。
    
2.  The sum of the eigenvalues (counted with multiplicity) equals the trace of $A$:
    特征值（按重数计算）的和等于 $A$ 的迹：
    
    $$
    \text{tr}(A) = \lambda_1 + \cdots + \lambda_n.
    $$
    
3.  The product of the eigenvalues equals the determinant of $A$:
    特征值的乘积等于 $A$ 的行列式：
    
    $$
    \det(A) = \lambda_1 \cdots \lambda_n.
    $$
    
4.  Similar matrices have the same characteristic polynomial, hence the same eigenvalues.
    相似的矩阵具有相同的特征多项式，因此具有相同的特征值。
    

### Geometric Interpretation
几何解释

The characteristic polynomial captures when $A - \lambda I$ collapses space: its determinant is zero precisely when the transformation $A - \lambda I$ is singular. Thus, eigenvalues mark the critical scalings where the matrix loses invertibility.
特征多项式捕捉了 $A - \lambda I$ 何时使空间坍缩：当变换 $A - \lambda I$ 为奇异时，其行列式恰好为零。因此，特征值标记了矩阵失去可逆性的临界尺度。

### Why this matters
为什么这很重要

Characteristic polynomials provide the computational tool to extract eigenvalues. They connect matrix invariants (trace and determinant) with geometry, and form the foundation for diagonalization, spectral theorems, and stability analysis in dynamical systems.
特征多项式提供了提取特征值的计算工具。它们将矩阵不变量（迹和行列式）与几何联系起来，并构成了动力系统中对角化、谱定理和稳定性分析的基础。

### Exercises 8.3
练习 8.3

1.  Compute the characteristic polynomial of
    计算特征多项式
    
    $$
    A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}.
    $$
    
2.  Verify that the sum of the eigenvalues of $\begin{bmatrix} 5 & 0 \\ 0 & -2 \end{bmatrix}$ equals its trace, and their product equals its determinant.
    验证特征值之和 $\begin{bmatrix} 5 & 0 \\ 0 & -2 \end{bmatrix}$ 等于它的迹，它们的乘积等于它的行列式。
    
3.  Show that for any triangular matrix, the eigenvalues are just the diagonal entries.
    证明对于任何三角矩阵，特征值只是对角线项。
    
4.  Prove that if $A$ and $B$ are similar matrices, then $p_A(\lambda) = p_B(\lambda)$.
    证明如果 $A$ 和 $B$ 是相似矩阵，则 $p_A(\lambda) = p_B(\lambda)$ 。
    
5.  Compute the characteristic polynomial of $\begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{bmatrix}$.
    计算特征多项式 \[ 1 1 0 0 1 1 0 0 1 \] ​ 1 0 0 ​ 1 1 0 ​ 0 1 1 ​ ​ .
    

## 8.4 Applications (Differential Equations, Markov Chains)
8.4 应用（微分方程、马尔可夫链）

Eigenvalues and eigenvectors are not only central to the theory of linear algebra-they are indispensable tools across mathematics and applied science. Two classic applications are solving systems of differential equations and analyzing Markov chains.
特征值和特征向量不仅是线性代数理论的核心，也是数学和应用科学领域中不可或缺的工具。两个经典的应用是求解微分方程组和分析马尔可夫链。

### Linear Differential Equations
线性微分方程

Consider the system
考虑系统

$$
\frac{d\mathbf{x}}{dt} = A \mathbf{x},
$$

where $A$ is an $n \times n$ matrix and $\mathbf{x}(t)$ is a vector-valued function.
其中 $A$ 是 $n \times n$ 矩阵， $\mathbf{x}(t)$ 是矢量值函数。

If $\mathbf{v}$ is an eigenvector of $A$ with eigenvalue $\lambda$, then the function
如果 $\mathbf{v}$ 是 $A$ 的特征向量，其特征值为 $\lambda$ ，则函数

$$
\mathbf{x}(t) = e^{\lambda t}\mathbf{v}
$$

is a solution.
是一个解决方案。

*   Eigenvalues determine the growth or decay rate:
    特征值决定增长率或衰减率：
    
    *   If $\lambda < 0$, solutions decay (stable).
        如果 $\lambda < 0$ ，则解决方案衰减（稳定）。
    *   If $\lambda > 0$, solutions grow (unstable).
        如果 $\lambda > 0$ ，则解决方案会增长（不稳定）。
    *   If $\lambda$ is complex, oscillations occur.
        如果 $\lambda$ 是复数，则会发生振荡。

By combining eigenvector solutions, we can solve general initial conditions.
通过结合特征向量解，我们可以解决一般的初始条件。

Example 8.4.1. Let
例 8.4.1. 设

$$
A = \begin{bmatrix}2 & 0 \\0 & -1 \end{bmatrix}.
$$

Then eigenvalues are $2, -1$with eigenvectors$(1,0)$, $(0,1)$. Solutions are
则特征值为 $2, -1 $with eigenvectors$ (1,0) $, $ (0,1)$。解为

$$
\mathbf{x}(t) = c_1 e^{2t}(1,0) + c_2 e^{-t}(0,1).
$$

Thus one component grows exponentially, the other decays.
因此，一个部分呈指数增长，另一个部分则衰减。

### Markov Chains
马尔可夫链

A Markov chain is described by a stochastic matrix $P$, where each column sums to 1 and entries are nonnegative. If $\mathbf{x}_k$ represents the probability distribution after $k$ steps, then
马尔可夫链可以用随机矩阵 $P$ 来描述，其中每列和为 1，且元素为非负值。如果 𝑥 𝑘 x k ​ 表示 $k$ 步后的概率分布，则

$$
\mathbf{x}_{k+1} = P \mathbf{x}_k.
$$

Iterating gives
迭代得到

$$
\mathbf{x}_k = P^k \mathbf{x}_0.
$$

Understanding long-term behavior reduces to analyzing powers of $P$.
理解长期行为可以归结为分析 $P$ 的力量。

*   The eigenvalue $\lambda = 1$ always exists. Its eigenvector gives the steady-state distribution.
    特征值 $\lambda = 1$ 始终存在。其特征向量给出了稳态分布。
*   All other eigenvalues satisfy $|\lambda| \leq 1$. Their influence decays as $k \to \infty$.
    所有其他特征值都满足 $|\lambda| \leq 1$ 。它们的影响衰减为 $k \to \infty$ 。

Example 8.4.2. Consider
例 8.4.2. 考虑

$$
P = \begin{bmatrix}0.9 & 0.5 \\0.1 & 0.5 \end{bmatrix}.
$$

Eigenvalues are $\lambda_1 = 1$, $\lambda_2 = 0.4$. The eigenvector for $\lambda = 1$ is proportional to $(5,1)$. Normalizing gives the steady state
特征值为 $\lambda_1 = 1$ , $\lambda_2 = 0.4$ 。 $\lambda = 1$ 的特征向量与 $(5,1)$ 成正比。归一化后可得到稳态

$$
\pi = \left(\tfrac{5}{6}, \tfrac{1}{6}\right).
$$

Thus, regardless of the starting distribution, the chain converges to $\pi$.
因此，无论起始分布如何，链都会收敛到 $\pi$ 。

### Geometric Interpretation
几何解释

*   In differential equations, eigenvalues determine the time evolution: exponential growth, decay, or oscillation.
    在微分方程中，特征值决定时间的演变：指数增长、衰减或振荡。
*   In Markov chains, eigenvalues determine the long-term equilibrium of stochastic processes.
    在马尔可夫链中，特征值决定了随机过程的长期均衡。

### Why this matters
为什么这很重要

Eigenvalue methods turn complex iterative or dynamical systems into tractable problems. In physics, engineering, and finance, they describe stability and resonance. In computer science and statistics, they power algorithms from Google’s PageRank to modern machine learning.
特征值方法将复杂的迭代或动态系统转化为易于处理的问题。在物理学、工程学和金融学领域，它们描述稳定性和共振。在计算机科学和统计学领域，它们为从谷歌的 PageRank 到现代机器学习等各种算法提供支持。

### Exercises 8.4
练习 8.4

1.  Solve $\tfrac{d}{dt}\mathbf{x} = \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix}\mathbf{x}$.
    解出 $\tfrac{d}{dt}\mathbf{x} = \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix}\mathbf{x}$ 。
    
2.  Show that if $A$ has a complex eigenvalue $\alpha \pm i\beta$, then solutions of $\tfrac{d}{dt}\mathbf{x} = A\mathbf{x}$ involve oscillations of frequency $\beta$.
    证明如果 $A$ 具有复特征值 $\alpha \pm i\beta$ ，则 $\tfrac{d}{dt}\mathbf{x} = A\mathbf{x}$ 的解涉及频率 $\beta$ 的振荡。
    
3.  Find the steady-state distribution of
    找到稳态分布
    
    $$
    P = \begin{bmatrix} 0.7 & 0.2 \\ 0.3 & 0.8 \end{bmatrix}.
    $$
    
4.  Prove that for any stochastic matrix $P$, 1 is always an eigenvalue.
    证明对于任何随机矩阵 $P$ ， 1 始终是特征值。
    
5.  Explain why all eigenvalues of a stochastic matrix satisfy $|\lambda| \leq 1$.
    解释为什么随机矩阵的所有特征值都满足 $|\lambda| \leq 1$ 。
    

# Chapter 9. Quadratic Forms and Spectral Theorems
第九章二次型和谱定理

## 9.1 Quadratic Forms
9.1 二次型

A quadratic form is a polynomial of degree two in several variables, expressed neatly using matrices. Quadratic forms appear throughout mathematics: in optimization, geometry of conic sections, statistics (variance), and physics (energy functions).
二次型是多元二次多项式，可以用矩阵简洁地表示。二次型在数学中随处可见：优化、圆锥曲线几何、统计学（方差）和物理学（能量函数）。

### Definition
定义

Let $A$ be an $n \times n$ symmetric matrix and $\mathbf{x} \in \mathbb{R}^n$. The quadratic form associated with $A$ is
令 $A$ 为 $n \times n$ 对称矩阵， $\mathbf{x} \in \mathbb{R}^n$ 。与 $A$ 相关的二次式为

$$
Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}.
$$

Expanded,
扩展，

$$
Q(\mathbf{x}) = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j.
$$

Because $A$ is symmetric ($a_{ij} = a_{ji}$), the cross-terms can be grouped naturally.
因为 $A$ 是对称的 (𝑎 𝑖 𝑗 = 𝑎 𝑗 𝑖 a 伊奇 ​ =a 姬 ​ )，交叉项可以自然分组。

### Examples
示例

Example 9.1.1. For
例 9.1.1. 对于

$$
A = \begin{bmatrix}2 & 1 \\1 & 3 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix}x \\y \end{bmatrix},
$$

 

$$
Q(x,y) = \begin{bmatrix} x & y \end{bmatrix}\begin{bmatrix}2 & 1 \\1 & 3 \end{bmatrix}\begin{bmatrix}x \\y \end{bmatrix}= 2x^2 + 2xy + 3y^2.
$$

Example 9.1.2. The quadratic form
例 9.1.2. 二次型

$$
Q(x,y) = x^2 + y^2
$$

corresponds to the matrix $A = I_2$. It measures squared Euclidean distance from the origin.
对应于矩阵𝐴 = 𝐼 2 A=I 2 ​ . 它测量距离原点的平方欧几里得距离。

Example 9.1.3. The conic section equation
例 9.1.3 圆锥曲线方程

$$
4x^2 + 2xy + 5y^2 = 1
$$

is described by the quadratic form $\mathbf{x}^T A \mathbf{x} = 1$ with
由二次型 $\mathbf{x}^T A \mathbf{x} = 1$ 描述

$$
A = \begin{bmatrix}4 & 1 \\1 & 5\end{bmatrix}.
$$

### Diagonalization of Quadratic Forms
二次型的对角化

By choosing a new basis consisting of eigenvectors of $A$, we can rewrite the quadratic form without cross terms. If $A = PDP^{-1}$ with $D$ diagonal, then
通过选择由 $A$ 的特征向量组成的新基，我们可以重写没有交叉项的二次型。如果 $A = PDP^{-1}$ 以 $D$ 为对角线，则

$$
Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = (P^{-1}\mathbf{x})^T D (P^{-1}\mathbf{x}).
$$

Thus quadratic forms can always be expressed as a sum of weighted squares:
因此二次型总是可以表示为加权平方和：

$$
Q(\mathbf{y}) = \lambda_1 y_1^2 + \cdots + \lambda_n y_n^2,
$$

where $\lambda_i$ are the eigenvalues of $A$.
其中𝜆 𝑖 λ i ​ 是 $A$ 的特征值。

### Geometric Interpretation
几何解释

Quadratic forms describe geometric shapes:
二次型描述几何形状：

*   In 2D: ellipses, parabolas, hyperbolas.
    二维：椭圆、抛物线、双曲线。
*   In 3D: ellipsoids, paraboloids, hyperboloids.
    在 3D 中：椭圆体、抛物面、双曲面。
*   In higher dimensions: generalizations of ellipsoids.
    在更高维度中：椭圆体的概括。

Diagonalization aligns the coordinate axes with the principal axes of the shape.
对角化将坐标轴与形状的主轴对齐。

### Why this matters
为什么这很重要

Quadratic forms unify geometry and algebra. They are central in optimization (minimizing energy functions), statistics ( covariance matrices and variance), mechanics (kinetic energy), and numerical analysis. Understanding quadratic forms leads directly to the spectral theorem.
二次型统一了几何和代数。它们在优化（最小化能量函数）、统计学（协方差矩阵和方差）、力学（动能）和数值分析中都至关重要。理解二次型可以直接引出谱定理。

### Exercises 9.1
练习 9.1

1.  Write the quadratic form $Q(x,y) = 3x^2 + 4xy + y^2$ as $\mathbf{x}^T A \mathbf{x}$ for some symmetric matrix $A$.
    对于某些对称矩阵 $A$ ，将二次型 $Q(x,y) = 3x^2 + 4xy + y^2$ 写为 $\mathbf{x}^T A \mathbf{x}$ 。
2.  For $A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$, compute $Q(x,y)$ explicitly.
    对于 $A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$ ，明确计算 $Q(x,y)$ 。
3.  Diagonalize the quadratic form $Q(x,y) = 2x^2 + 2xy + 3y^2$.
    将二次型 $Q(x,y) = 2x^2 + 2xy + 3y^2$ 对角化。
4.  Identify the conic section given by $Q(x,y) = x^2 - y^2$.
    确定由 $Q(x,y) = x^2 - y^2$ 给出的圆锥截面。
5.  Show that if $A$ is symmetric, quadratic forms defined by $A$ and $A^T$ are identical.
    证明如果 $A$ 是对称的，则由 $A$ 和 $A^T$ 定义的二次型是相同的。

## 9.2 Positive Definite Matrices
9.2 正定矩阵

Quadratic forms are especially important when their associated matrices are positive definite, since these guarantee positivity of energy, distance, or variance. Positive definiteness is a cornerstone in optimization, numerical analysis, and statistics.
当二次型的相关矩阵为正定矩阵时，它们尤为重要，因为它们可以保证能量、距离或方差的正性。正定性是优化、数值分析和统计学的基石。

### Definition
定义

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is called:
对称矩阵 $A \in \mathbb{R}^{n \times n}$ 称为：

*   Positive definite if
    正定如果
    
    $$
    \mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all nonzero } \mathbf{x} \in \mathbb{R}^n.
    $$
    
*   Positive semidefinite if
    正半定的，如果
    
    $$
    \mathbf{x}^T A \mathbf{x} \geq 0 \quad \text{for all } \mathbf{x}.
    $$
    

Similarly, negative definite (always < 0) and indefinite (can be both < 0 and > 0) matrices are defined.
类似地，定义了负定（始终 < 0）和不定（可以同时 < 0 和 > 0）矩阵。

### Examples
示例

Example 9.2.1.
例 9.2.1。

$$
A = \begin{bmatrix}2 & 0 \\0 & 3 \end{bmatrix}
$$

is positive definite, since
是正定的，因为

$$
Q(x,y) = 2x^2 + 3y^2 > 0
$$

for all $(x,y) \neq (0,0)$.
对于所有 $(x,y) \neq (0,0)$ 。

Example 9.2.2.
例 9.2.2。

$$
A = \begin{bmatrix}1 & 2 \\2 & 1 \end{bmatrix}
$$

has quadratic form
具有二次形式

$$
Q(x,y) = x^2 + 4xy + y^2.
$$

This matrix is not positive definite, since $Q(1,-1) = -2 < 0$.
该矩阵不是正定的，因为 $Q(1,-1) = -2 < 0$ 。

### Characterizations
特征

For a symmetric matrix $A$:
对于对称矩阵 $A$ ：

1.  Eigenvalue test: $A$ is positive definite if and only if all eigenvalues of $A$ are positive.
    特征值检验：当且仅当 $A$ 的所有特征值都为正时， $A$ 才是正定的。
    
2.  Principal minors test (Sylvester’s criterion): $A$ is positive definite if and only if all leading principal minors ( determinants of top-left $k \times k$ submatrices) are positive.
    主子式检验（西尔维斯特标准）：当且仅当所有首项主子式（左上角 $k \times k$ 子矩阵的行列式）均为正时， $A$ 才是正定的。
    
3.  Cholesky factorization: $A$ is positive definite if and only if it can be written as
    Cholesky 分解： $A$ 为正定当且仅当它可以写成
    
    $$
    A = R^T R,
    $$
    
    where $R$ is an upper triangular matrix with positive diagonal entries.
    其中 $R$ 是具有正对角线项的上三角矩阵。
    

### Geometric Interpretation
几何解释

*   Positive definite matrices correspond to quadratic forms that define ellipsoids centered at the origin.
    正定矩阵对应于定义以原点为中心的椭圆体的二次型。
*   Positive semidefinite matrices define flattened ellipsoids (possibly degenerate).
    正半定矩阵定义扁平的椭球体（可能是退化的）。
*   Indefinite matrices define hyperbolas or saddle-shaped surfaces.
    不定矩阵定义双曲线或马鞍形曲面。

### Applications
应用

*   Optimization: Hessians of convex functions are positive semidefinite; strict convexity corresponds to positive definite Hessians.
    优化：凸函数的 Hessian 矩阵是正半定的；严格凸性对应于正定的 Hessian 矩阵。
*   Statistics: Covariance matrices are positive semidefinite.
    统计：协方差矩阵是正半定的。
*   Numerical methods: Cholesky decomposition is widely used to solve systems with positive definite matrices efficiently.
    数值方法：Cholesky 分解被广泛用于有效地解决具有正定矩阵的系统。

### Why this matters
为什么这很重要

Positive definiteness provides stability and guarantees in mathematics and computation. It ensures energy functions are bounded below, optimization problems have unique solutions, and statistical models are meaningful.
正定性在数学和计算中提供了稳定性和保证。它确保能量函数有界，优化问题有唯一解，统计模型有意义。

### Exercises 9.2
练习 9.2

1.  Use Sylvester’s criterion to check whether
    使用 Sylvester 标准检查
    
    $$
    A = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}
    $$
    
    is positive definite.
    是正定的。
    
2.  Determine whether
    确定是否
    
    $$
    A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
    $$
    
    is positive definite, semidefinite, or indefinite.
    是正定的、半定的或不定的。
    
3.  Find the eigenvalues of
    找到特征值
    
    $$
    A = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix},
    $$
    
    and use them to classify definiteness.
    并用它们来对确定性进行分类。
    
4.  Prove that all diagonal matrices with positive entries are positive definite.
    证明所有具有正项的对角矩阵都是正定的。
    
5.  Show that if $A$ is positive definite, then so is $P^T A P$ for any invertible matrix $P$.
    证明如果 $A$ 为正定矩阵，则对于任何可逆矩阵 $P$ ， $P^T A P$ 也为正定矩阵。
    

## 9.3 Spectral Theorem
9.3 谱定理

The spectral theorem is one of the most powerful results in linear algebra. It states that symmetric matrices can always be diagonalized by an orthogonal basis of eigenvectors. This links algebra (eigenvalues), geometry (orthogonal directions), and applications (stability, optimization, statistics).
谱定理是线性代数中最有力的结论之一。它指出对称矩阵总是可以通过特征向量的正交基对角化。这连接了代数（特征值）、几何（正交方向）和应用（稳定性、优化、统计）。

### Statement of the Spectral Theorem
谱定理表述

If $A \in \mathbb{R}^{n \times n}$ is symmetric ($A^T = A$), then:
如果 $A \in \mathbb{R}^{n \times n}$ 是对称的（ $A^T = A$ ），则：

1.  All eigenvalues of $A$ are real.
    $A$ 的所有特征值都是实数。
    
2.  There exists an orthonormal basis of $\mathbb{R}^n$ consisting of eigenvectors of $A$.
    存在由 $A$ 的特征向量组成的 $\mathbb{R}^n$ 正交基。
    
3.  Thus, $A$ can be written as
    因此， $A$ 可以写成
    
    $$
    A = Q \Lambda Q^T,
    $$
    
    where $Q$ is an orthogonal matrix ($Q^T Q = I$) and $\Lambda$ is diagonal with eigenvalues of $A$ on the diagonal.
    其中 $Q$ 是正交矩阵 ( $Q^T Q = I$ )， $\Lambda$ 是对角矩阵，其特征值 $A$ 位于对角线上。
    

### Consequences
结果

*   Symmetric matrices are always diagonalizable, and the diagonalization is numerically stable.
    对称矩阵总是可对角化的，并且对角化在数值上是稳定的。
*   Quadratic forms $\mathbf{x}^T A \mathbf{x}$ can be expressed in terms of eigenvalues and eigenvectors, showing ellipsoids aligned with eigen-directions.
    二次型 $\mathbf{x}^T A \mathbf{x}$ 可以用特征值和特征向量来表示，显示与特征方向对齐的椭圆体。
*   Positive definiteness can be checked by confirming that all eigenvalues are positive.
    可以通过确认所有特征值都为正来检查正定性。

### Example 9.3.1
例 9.3.1

Let
让

$$
A = \begin{bmatrix}2 & 1 \\1 & 2 \end{bmatrix}.
$$

1.  Characteristic polynomial:
    特征多项式：

$$
p(\lambda) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3.
$$

Eigenvalues: $\lambda_1 = 1, \ \lambda_2 = 3$.
特征值： $\lambda_1 = 1, \ \lambda_2 = 3$ 。

2.  Eigenvectors:
    特征向量：

*   For $\lambda=1$: solve $(A-I)\mathbf{v} = 0$, giving $(1,-1)$.
    对于 $\lambda=1$ ：求解 $(A-I)\mathbf{v} = 0$ ，得到 $(1,-1)$ 。
*   For $\lambda=3$: solve $(A-3I)\mathbf{v} = 0$, giving $(1,1)$.
    对于 $\lambda=3$ ：求解 $(A-3I)\mathbf{v} = 0$ ，得到 $(1,1)$ 。

3.  Normalize eigenvectors:
    归一化特征向量：

$$
\mathbf{u}_1 = \tfrac{1}{\sqrt{2}}(1,-1), \quad \mathbf{u}_2 = \tfrac{1}{\sqrt{2}}(1,1).
$$

4.  Then
    然后

$$
Q =\begin{bmatrix}\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} \[6pt] -\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}\end{bmatrix}, \quad\Lambda =\begin{bmatrix}1 & 0 \\0 & 3\end{bmatrix}.
$$

So
所以

$$
A = Q \Lambda Q^T.
$$

### Geometric Interpretation
几何解释

The spectral theorem says every symmetric matrix acts like independent scaling along orthogonal directions. In geometry, this corresponds to stretching space along perpendicular axes.
谱定理指出，每个对称矩阵都像沿正交方向的独立缩放一样。在几何学中，这相当于沿垂直轴拉伸空间。

*   Ellipses, ellipsoids, and quadratic surfaces can be fully understood via eigenvalues and eigenvectors.
    通过特征值和特征向量可以充分理解椭圆、椭圆体和二次曲面。
*   Orthogonality ensures directions remain perpendicular after transformation.
    正交性确保方向在变换后保持垂直。

### Applications
应用

*   Optimization: The spectral theorem underlies classification of critical points via eigenvalues of the Hessian.
    优化：谱定理是通过 Hessian 的特征值对临界点进行分类的基础。
*   PCA (Principal Component Analysis): Data covariance matrices are symmetric, and PCA finds orthogonal directions of maximum variance.
    PCA（主成分分析）：数据协方差矩阵是对称的，PCA 找到最大方差的正交方向。
*   Differential equations & physics: Symmetric operators correspond to measurable quantities with real eigenvalues ( stability, energy).
    微分方程和物理学：对称算子对应于具有实特征值（稳定性、能量）的可测量量。

### Why this matters
为什么这很重要

The spectral theorem guarantees that symmetric matrices are as simple as possible: they can always be analyzed in terms of real, orthogonal eigenvectors. This provides both deep theoretical insight and powerful computational tools.
谱定理保证对称矩阵尽可能简单：它们总是可以用实数正交特征向量来分析。这既提供了深刻的理论见解，也提供了强大的计算工具。

### Exercises 9.3
练习 9.3

1.  Diagonalize
    对角化
    
    $$
    A = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}
    $$
    
    using the spectral theorem.
    使用谱定理。
    
2.  Prove that all eigenvalues of a real symmetric matrix are real.
    证明实对称矩阵的所有特征值都是实数。
    
3.  Show that eigenvectors corresponding to distinct eigenvalues of a symmetric matrix are orthogonal.
    证明对称矩阵的不同特征值对应的特征向量是正交的。
    
4.  Explain geometrically how the spectral theorem describes ellipsoids defined by quadratic forms.
    从几何角度解释谱定理如何描述由二次型定义的椭球体。
    
5.  Apply the spectral theorem to the covariance matrix
    将谱定理应用于协方差矩阵
    
    $$
    \Sigma = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix},
    $$
    
    and interpret the eigenvectors as principal directions of variance.
    并将特征向量解释为方差的主方向。
    

## 9.4 Principal Component Analysis (PCA)
9.4 主成分分析（PCA）

Principal Component Analysis (PCA) is a widely used technique in data science, machine learning, and statistics. At its core, PCA is an application of the spectral theorem to covariance matrices: it finds orthogonal directions (principal components) that capture the maximum variance in data.
主成分分析 (PCA) 是数据科学、机器学习和统计学中广泛使用的技术。PCA 的核心是谱定理在协方差矩阵中的应用：它找到能够捕捉数据中最大方差的正交方向（主成分）。

### The Idea
理念

Given a dataset of vectors $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_m \in \mathbb{R}^n$:
给定向量数据集 $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_m \in \mathbb{R}^n$ ：

1.  Center the data by subtracting the mean vector $\bar{\mathbf{x}}$.
    通过减去平均向量 $\bar{\mathbf{x}}$ 使数据居中。
    
2.  Form the covariance matrix
    形成协方差矩阵
    
    $$
    \Sigma = \frac{1}{m} \sum_{i=1}^m (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T.
    $$
    
3.  Apply the spectral theorem: $\Sigma = Q \Lambda Q^T$.
    应用谱定理： $\Sigma = Q \Lambda Q^T$ 。
    
    *   Columns of $Q$ are orthonormal eigenvectors (principal directions).
        $Q$ 的列是正交特征向量（主方向）。
    *   Eigenvalues in $\Lambda$ measure variance explained by each direction.
        $\Lambda$ 中的特征值测量每个方向解释的方差。

The first principal component is the eigenvector corresponding to the largest eigenvalue; it is the direction of maximum variance.
第一个主成分是最大特征值对应的特征向量，是方差最大的方向。

### Example 9.4.1
例 9.4.1

Suppose we have two-dimensional data points roughly aligned along the line $y = x$. The covariance matrix is approximately
假设我们有二维数据点大致沿着直线 $y = x$ 排列。协方差矩阵大约为

$$
\Sigma =\begin{bmatrix}2 & 1.9 \\1.9 & 2\end{bmatrix}.
$$

Eigenvalues are about $3.9 and \\0.1$. The eigenvector for $\\lambda = 3.9$is approximately$(1,1)/\\sqrt{2}$.
特征值约为 $3.9 和 \\ 0.1 $. The eigenvector for $ \\lambda = 3.9 $is approximately$ (1,1)/\\sqrt{2}$。

*   First principal component: the line $y = x$.
    第一个主成分：线 $y = x$ 。
*   Most variance lies along this direction.
    大部分差异都发生在这个方向。
*   Second component is nearly orthogonal ($y = -x$), but variance there is tiny.
    第二个成分几乎正交（ $y = -x$ ），但那里的方差很小。

Thus PCA reduces the data to essentially one dimension.
因此，PCA 将数据简化为一个维度。

### Applications of PCA
PCA 的应用

1.  Dimensionality reduction: Represent data with fewer features while retaining most variance.
    降维：用较少的特征表示数据，同时保留大部分的方差。
2.  Noise reduction: Small eigenvalues correspond to noise; discarding them filters data.
    降噪：较小的特征值对应噪声；丢弃它们可以过滤数据。
3.  Visualization: Projecting high-dimensional data onto top 2 or 3 principal components reveals structure.
    可视化：将高维数据投影到前 2 个或 3 个主成分上可以揭示结构。
4.  Compression: PCA is used in image and signal compression.
    压缩：PCA 用于图像和信号压缩。

### Connection to the Spectral Theorem
与谱定理的联系

The covariance matrix $\Sigma$ is always symmetric and positive semidefinite. Hence by the spectral theorem, it has an orthonormal basis of eigenvectors and nonnegative real eigenvalues. PCA is nothing more than re-expressing data in this eigenbasis.
协方差矩阵 $\Sigma$ 始终是对称的，且为半正定矩阵。因此，根据谱定理，它有一个由特征向量和非负实特征值组成的正交基。PCA 只不过是在这个特征基上重新表达数据。

### Why this matters
为什么这很重要

PCA demonstrates how abstract linear algebra directly powers modern applications. Eigenvalues and eigenvectors give a practical method for simplifying data, revealing patterns, and reducing complexity. It is one of the most important algorithms derived from the spectral theorem.
PCA 展示了抽象线性代数如何直接驱动现代应用。特征值和特征向量提供了一种简化数据、揭示模式和降低复杂性的实用方法。它是从谱定理中推导出的最重要的算法之一。

### Exercises 9.4
练习 9.4

1.  Show that the covariance matrix is symmetric and positive semidefinite.
    证明协方差矩阵是对称的和半正定的。
2.  Compute the covariance matrix of the dataset $(1,2), (2,3), (3,4)$, and find its eigenvalues and eigenvectors.
    计算数据集 $(1,2), (2,3), (3,4)$ 的协方差矩阵，并找到其特征值和特征向量。
3.  Explain why the first principal component captures the maximum variance.
    解释为什么第一个主成分捕获最大方差。
4.  In image compression, explain how PCA can reduce storage by keeping only the top $k$ principal components.
    在图像压缩中，解释 PCA 如何通过仅保留前 $k$ 个主成分来减少存储。
5.  Prove that the sum of the eigenvalues of the covariance matrix equals the total variance of the dataset.
    证明协方差矩阵的特征值之和等于数据集的总方差。

# Chapter 10. Linear Algebra in Practice
第 10 章 线性代数实践

## 10.1 Computer Graphics (Rotations, Projections)
10.1 计算机图形学（旋转、投影）

Linear algebra is the language of modern computer graphics. Every image rendered on a screen, every 3D model rotated or projected, is ultimately the result of applying matrices to vectors. Rotations, reflections, scalings, and projections are all linear transformations, making matrices the natural tool for manipulating geometry.
线性代数是现代计算机图形学的语言。屏幕上渲染的每一幅图像，以及旋转或投影的每一个 3D 模型，最终都是将矩阵应用于向量的结果。旋转、反射、缩放和投影都是线性变换，这使得矩阵成为处理几何图形的天然工具。

### Rotations in 2D
二维旋转

A counterclockwise rotation by an angle $\theta$ in the plane is represented by
在平面上逆时针旋转角度 $\theta$ 表示为

$$
R_\theta =\begin{bmatrix}\cos\theta & -\sin\theta \\\sin\theta & \cos\theta\end{bmatrix}.
$$

For any vector $\mathbf{v} \in \mathbb{R}^2$, the rotated vector is
对于任意向量 $\mathbf{v} \in \mathbb{R}^2$ ，旋转后的向量为

$$
\mathbf{v}' = R_\theta \mathbf{v}.
$$

This preserves lengths and angles, since $R_\theta$ is orthogonal with determinant 1.
这保留了长度和角度，因为𝑅 𝜃 R θ ​ 与行列式 1 正交。

### Rotations in 3D
3D 旋转

In three dimensions, rotations are represented by $3 \\times 3 orthogonal matrices with determinant \\1$. For example, arotation about the $z$-axis is
在三维空间中，旋转由$3 \\times 3 正交矩阵表示，其行列式为 \\ 1 $. For example, arotation about the $ z$轴为

$$
R_z(\theta) =\begin{bmatrix}\cos\theta & -\sin\theta & 0 \\\sin\theta & \cos\theta & 0 \\0 & 0 & 1\end{bmatrix}.
$$

Similar formulas exist for rotations about the $x$\- and $y$\-axes.
对于绕 $x$ 轴和 $y$ 轴的旋转也存在类似的公式。

More general 3D rotations can be described by axis–angle representation or quaternions, but the underlying idea is still linear transformations represented by matrices.
更一般的 3D 旋转可以用轴角表示或四元数来描述，但其基本思想仍然是矩阵表示的线性变换。

### Projections
预测

To display 3D objects on a 2D screen, we use projections:
为了在 2D 屏幕上显示 3D 对象，我们使用投影：

1.  Orthogonal projection: drops the $z$\-coordinate, mapping $(x,y,z) \mapsto (x,y)$.
    正交投影：删除 $z$ 坐标，映射 $(x,y,z) \mapsto (x,y)$ 。
    
    $$
    P = \begin{bmatrix}1 & 0 & 0 \\0 & 1 & 0\end{bmatrix}.
    $$
    
2.  Perspective projection: mimics the effect of a camera. A point $(x,y,z)$ projects to
    透视投影：模拟相机的效果。点 $(x,y,z)$ 投影到
    
    $$
    \left(\frac{x}{z}, \frac{y}{z}\right),
    $$
    
    capturing how distant objects appear smaller.
    捕捉远处物体如何显得更小。
    

These operations are linear (orthogonal projection) or nearly linear (perspective projection becomes linear in homogeneous coordinates).
这些操作是线性的（正交投影）或近似线性的（透视投影在齐次坐标中变为线性）。

### Homogeneous Coordinates
齐次坐标

To unify translations and projections with linear transformations, computer graphics uses homogeneous coordinates. A 3D point $(x,y,z)$ is represented as a 4D vector $(x,y,z,1)$. Transformations are then 4×4 matrices, which can represent rotations, scalings, and translations in a single framework.
为了将平移和投影与线性变换统一起来，计算机图形学使用齐次坐标。3D 点 $(x,y,z)$ 表示为四维向量 $(x,y,z,1)$ 。变换则表示为矩阵 4×4 ，可以在单个框架中表示旋转、缩放和平移。

Example: Translation by $(a,b,c)$:
例如： $(a,b,c)$ 翻译：

$$
T = \begin{bmatrix}1 & 0 & 0 & a \\0 & 1 & 0 & b \\0 & 0 & 1 & c \\0 & 0 & 0 & 1\end{bmatrix}.
$$

### Geometric Interpretation
几何解释

*   Rotations preserve shape and size, only changing orientation.
    旋转保持形状和大小，仅改变方向。
*   Projections reduce dimension: from 3D world space to 2D screen space.
    投影减少维度：从 3D 世界空间到 2D 屏幕空间。
*   Homogeneous coordinates allow us to combine multiple transformations (rotation + translation + projection) into a single matrix multiplication.
    齐次坐标允许我们将多个变换（旋转+平移+投影）组合成单个矩阵乘法。

### Why this matters
为什么这很重要

Linear algebra enables all real-time graphics: video games, simulations, CAD software, and movie effects. By chaining simple matrix operations, complex transformations are applied efficiently to millions of points per second.
线性代数支持所有实时图形：视频游戏、模拟、CAD 软件和电影特效。通过链接简单的矩阵运算，复杂的变换可以高效地应用于每秒数百万个点。

### Exercises 10.1
练习10.1

1.  Write the rotation matrix for a 90° counterclockwise rotation in $\mathbb{R}^2$. Apply it to $(1,0)$.
    在 $\mathbb{R}^2$ 中写出逆时针旋转 90° 的旋转矩阵。将其应用到 $(1,0)$ 。
2.  Rotate the point $(1,1,0)$ about the $z$\-axis by 180°.
    将点 $(1,1,0)$ 绕 $z$ 轴旋转 180°。
3.  Show that the determinant of any 2D or 3D rotation matrix is 1.
    证明任何二维或三维旋转矩阵的行列式为 1。
4.  Derive the orthogonal projection matrix from $\mathbb{R}^3$ to the $xy$\-plane.
    推导从 $\mathbb{R}^3$ 到 $xy$ 平面的正交投影矩阵。
5.  Explain how homogeneous coordinates allow translations to be represented as matrix multiplications.
    解释齐次坐标如何允许平移表示为矩阵乘法。

## 10.2 Data Science (Dimensionality Reduction, Least Squares)
10.2 数据科学（降维、最小二乘）

Linear algebra provides the foundation for many data science techniques. Two of the most important are dimensionality reduction, where high-dimensional datasets are compressed while preserving essential information, and the least squares method, which underlies regression and model fitting.
线性代数为许多数据科学技术奠定了基础。其中最重要的两个技术是降维（在保留基本信息的同时压缩高维数据集）和最小二乘法（回归和模型拟合的基础）。

### Dimensionality Reduction
降维

High-dimensional data often contains redundancy: many features are correlated, meaning the data essentially lies near a lower-dimensional subspace. Dimensionality reduction identifies these subspaces.
高维数据通常包含冗余：许多特征相互关联，这意味着数据本质上位于低维子空间附近。降维可以识别这些子空间。

*   PCA (Principal Component Analysis): As introduced earlier, PCA diagonalizes the covariance matrix of the data.
    PCA（主成分分析）：如前所述，PCA 将数据的协方差矩阵对角化。
    
    *   Eigenvectors (principal components) define orthogonal directions of maximum variance.
        特征向量（主成分）定义最大方差的正交方向。
    *   Eigenvalues measure how much variance lies along each direction.
        特征值衡量每个方向上的方差。
    *   Keeping only the top $k$ components reduces data from $n$\-dimensional space to $k$\-dimensional space while retaining most variability.
        仅保留前 $k$ 个成分可将数据从 $n$ 维空间减少到 $k$ 维空间，同时保留大部分可变性。

Example 10.2.1. A dataset of 1000 images, each with 1024 pixels, may have most variance captured by just 50 eigenvectors of the covariance matrix. Projecting onto these components compresses the data while preserving essential features.
例 10.2.1。一个包含 1000 幅图像的数据集，每幅图像有 1024 个像素，其大部分方差可能仅由协方差矩阵的 50 个特征向量捕获。投影到这些分量上可以压缩数据，同时保留基本特征。

### Least Squares
最小二乘法

Often, we have more equations than unknowns-an overdetermined system:
通常，我们的方程比未知数还多——一个超定系统：

$$
A\mathbf{x} \approx \mathbf{b}, \quad A \in \mathbb{R}^{m \times n}, \ m > n.
$$

An exact solution may not exist. Instead, we seek $\mathbf{x}$ that minimizes the error
精确解可能不存在。因此，我们寻求最小化误差的 $\mathbf{x}$

$$
\|A\mathbf{x} - \mathbf{b}\|^2.
$$

This leads to the normal equations:
这导致了正规方程：

$$
A^T A \mathbf{x} = A^T \mathbf{b}.
$$

The solution is the orthogonal projection of $\mathbf{b}$ onto the column space of $A$.
解决方案是将 $\mathbf{b}$ 正交投影到 $A$ 的列空间上。

### Example 10.2.2
例 10.2.2

Fit a line $y = mx + c$ to data points $(x_i, y_i)$.
将线 $y = mx + c$ 与数据点 $(x_i, y_i)$ 拟合。

Matrix form:
矩阵形式：

$$
A = \begin{bmatrix}x_1 & 1 \\x_2 & 1 \\\vdots & \vdots \\x_m & 1\end{bmatrix},\quad\mathbf{b} =\begin{bmatrix}y_1 \\y_2 \\\vdots \\y_m \end{bmatrix},\quad\mathbf{x} =\begin{bmatrix}m \\c \end{bmatrix}.
$$

Solve $A^T A \mathbf{x} = A^T \mathbf{b}$. This yields the best-fit line in the least squares sense.
求解 $A^T A \mathbf{x} = A^T \mathbf{b}$ 。这将得出最小二乘意义上的最佳拟合线。

### Geometric Interpretation
几何解释

*   Dimensionality reduction: Find the best subspace capturing most variance.
    降维：找到捕获最多方差的最佳子空间。
*   Least squares: Project the target vector onto the subspace spanned by predictors.
    最小二乘：将目标向量投影到预测变量所跨越的子空间上。

Both are projection problems, solved using inner products and orthogonality.
两者都是投影问题，使用内积和正交性来解决。

### Why this matters
为什么这很重要

Dimensionality reduction makes large datasets tractable, filters noise, and reveals structure. Least squares fitting powers regression, statistics, and machine learning. Both rely directly on eigenvalues, eigenvectors, and projections-core tools of linear algebra.
降维使大型数据集更易于处理，过滤噪声并揭示结构。最小二乘拟合为回归、统计和机器学习提供支持。两者都直接依赖于特征值、特征向量和投影——线性代数的核心工具。

### Exercises 10.2
练习10.2

1.  Explain why PCA reduces noise in datasets by discarding small eigenvalue components.
    解释为什么 PCA 通过丢弃较小的特征值分量来减少数据集中的噪声。
2.  Compute the least squares solution to fitting a line through $(0,0), (1,1), (2,2)$.
    计算通过 $(0,0), (1,1), (2,2)$ 拟合直线的最小二乘解。
3.  Show that the least squares solution is unique if and only if $A^T A$ is invertible.
    证明最小二乘解是唯一的当且仅当 $A^T A$ 可逆。
4.  Prove that the least squares solution minimizes the squared error by projection arguments.
    证明最小二乘解通过投影参数最小化平方误差。
5.  Apply PCA to the data points $(1,0), (2,1), (3,2)$ and find the first principal component.
    将 PCA 应用于数据点 $(1,0), (2,1), (3,2)$ 并找到第一个主成分。

## 10.3 Networks and Markov Chains
10.3 网络和马尔可夫链

Graphs and networks provide a natural setting where linear algebra comes to life. From modeling flows and connectivity to predicting long-term behavior, matrices translate network structure into algebraic form. Markov chains, already introduced in Section 8.4, are a central example of networks evolving over time.
图和网络为线性代数的运用提供了自然的平台。从建模流和连接到预测长期行为，矩阵将网络结构转化为代数形式。马尔可夫链（已在 8.4 节介绍）是网络随时间演化的一个典型例子。

### Adjacency Matrices
邻接矩阵

A network (graph) with $n$ nodes can be represented by an adjacency matrix $A \in \mathbb{R}^{n \times n}$:
具有 $n$ 个节点的网络（图）可以用邻接矩阵 $A \in \mathbb{R}^{n \times n}$ 表示：

$$
A_{ij} =\begin{cases}1 & \text{if there is an edge from node \(i\) to node \(j\)} \\0 & \text{otherwise.}\end{cases}
$$

For weighted graphs, entries may be positive weights instead of 0/1.
对于加权图，条目可能是正权重而不是 0/1 。

*   The number of walks of length $k$ from node $i$ to node $j$ is given by the entry $(A^k)_{ij}$.
    从节点 $i$ 到节点 $j$ 的长度为 $k$ 的步行次数由条目 ( 𝐴 𝑘 ) 𝑖 𝑗 （一个 k ) 伊奇 ​ .
*   Powers of adjacency matrices thus encode connectivity over time.
    因此，邻接矩阵的幂可以对随时间变化的连通性进行编码。

### Laplacian Matrices
拉普拉斯矩阵

Another important matrix is the graph Laplacian:
另一个重要的矩阵是图拉普拉斯矩阵：

$$
L = D - A,
$$

where $D$ is the diagonal degree matrix ( $D_{ii} = \text{degree}(i)$ ).
其中 $D$ 是对角度矩阵 ( $D_{ii} = \text{degree}(i)$ )。

*   $L$ is symmetric and positive semidefinite.
    $L$ 是对称的并且是正半定的。
*   The smallest eigenvalue is always $0$, with eigenvector $(1,1,\\dots,1)$.
    最小特征值始终是 $0 $, with eigenvector $ (1,1,\\dots,1)$。
*   The multiplicity of eigenvalue 0 equals the number of connected components in the graph.
    特征值 0 的多重性等于图中连通分量的数量。

This connection between eigenvalues and connectivity forms the basis of spectral graph theory.
特征值和连通性之间的这种联系构成了谱图理论的基础。

### Markov Chains on Graphs
图上的马尔可夫链

A Markov chain can be viewed as a random walk on a graph. If $P$ is the transition matrix where $P_{ij}$ is the probability of moving from node $i$ to node $j$, then
马尔可夫链可以看作图上的随机游动。设 $P$ 为转移矩阵，其中 𝑃 𝑖 𝑗 P 伊奇 ​ 是从节点 $i$ 移动到节点 $j$ 的概率，那么

$$
\mathbf{x}_{k+1} = P \mathbf{x}_k
$$

describes the distribution of positions after $k$ steps.
描述 $k$ 步之后的位置分布。

*   The steady-state distribution is given by the eigenvector of $P$ with eigenvalue 1.
    稳态分布由特征向量 $P$ 给出，特征值为 1 。
*   The speed of convergence depends on the gap between the largest eigenvalue (which is always 1) and the second largest eigenvalue.
    收敛速度取决于最大特征值（始终为 1 ）与第二大特征值之间的差距。

### Example 10.3.1
例 10.3.1

Consider a simple 3-node cycle graph:
考虑一个简单的 3 节点循环图：

$$
P = \begin{bmatrix}0 & 1 & 0 \\0 & 0 & 1 \\1 & 0 & 0\end{bmatrix}.
$$

This Markov chain cycles deterministically among the nodes. Eigenvalues are the cube roots of unity: $1, e^{2\\pi i/3}, e^{4\\pi i/3}. The eigenvalue \\1$corresponds to the steady state, which is the uniformdistribution$(1/3,1/3,1/3)$.
这个马尔可夫链在节点之间确定性地循环。特征值是 统一：$1,e^{2\\pi i/3},e^{4\\pi i/3} 。特征值 \\ 1 $corresponds to the steady state, which is the uniformdistribution$ (1/3,1/3,1/3)$。

### Applications
应用

*   Search engines: Google’s PageRank algorithm models the web as a Markov chain, where steady-state probabilities rank pages.
    搜索引擎：Google 的 PageRank 算法将网络建模为马尔可夫链，其中稳态概率对网页进行排名。
*   Network analysis: Eigenvalues of adjacency or Laplacian matrices reveal communities, bottlenecks, and robustness.
    网络分析：邻接矩阵或拉普拉斯矩阵的特征值揭示社区、瓶颈和稳健性。
*   Epidemiology and information flow: Random walks model how diseases or ideas spread through networks.
    流行病学和信息流：随机游动模拟疾病或思想如何通过网络传播。

### Why this matters
为什么这很重要

Linear algebra transforms network problems into matrix problems. Eigenvalues and eigenvectors reveal connectivity, flow, stability, and long-term dynamics. Networks are everywhere-social media, biology, finance, and the internet-so these tools are indispensable.
线性代数将网络问题转化为矩阵问题。特征值和特征向量揭示了连通性、流动、稳定性和长期动态。网络无处不在——社交媒体、生物、金融和互联网——因此这些工具不可或缺。

### Exercises 10.3
练习10.3

1.  Write the adjacency matrix of a square graph with 4 nodes. Compute $A^2$ and interpret the entries.
    写出一个有 4 个节点的正方形图的邻接矩阵。计算 $A^2$ 并解释其中的元素。
    
2.  Show that the Laplacian of a connected graph has exactly one zero eigenvalue.
    证明连通图的拉普拉斯算子恰好有一个零特征值。
    
3.  Find the steady-state distribution of the Markov chain with
    找到马尔可夫链的稳态分布
    
    $$
    P = \begin{bmatrix} 0.5 & 0.5 \\ 0.4 & 0.6 \end{bmatrix}.
    $$
    
4.  Explain how eigenvalues of the Laplacian can detect disconnected components of a graph.
    解释拉普拉斯算子的特征值如何检测图中不连续的组成部分。
    
5.  Describe how PageRank modifies the transition matrix of the web graph to ensure a unique steady-state distribution.
    描述 PageRank 如何修改网络图的转换矩阵以确保唯一的稳态分布。
    

## 10.4 Machine Learning Connections
10.4 机器学习连接

Modern machine learning is built on linear algebra. From the representation of data as matrices to the optimization of large-scale models, nearly every step relies on concepts such as vector spaces, projections, eigenvalues, and matrix decompositions.
现代机器学习建立在线性代数的基础上。从数据矩阵表示到大规模模型的优化，几乎每一步都依赖于向量空间、投影、特征值和矩阵分解等概念。

### Data as Matrices
数据作为矩阵

A dataset with $m$ examples and $n$ features is represented as a matrix $X \in \mathbb{R}^{m \times n}$:
具有 $m$ 个示例和 $n$ 个特征的数据集表示为矩阵 $X \in \mathbb{R}^{m \times n}$ ：

$$
X =\begin{bmatrix}\- & \mathbf{x}_1^T & - \\\- & \mathbf{x}_2^T & - \\& \vdots & \\\- & \mathbf{x}_m^T & -\end{bmatrix},
$$

where each row $\mathbf{x}_i \in \mathbb{R}^n$ is a feature vector. Linear algebra provides tools to analyze, compress, and transform this data.
其中每行 $\mathbf{x}_i \in \mathbb{R}^n$ 是一个特征向量。线性代数提供了分析、压缩和转换此类数据的工具。

### Linear Models
线性模型

At the heart of machine learning are linear predictors:
机器学习的核心是线性预测器：

$\hat{y} = X\mathbf{w},$

where $\mathbf{w}$ is the weight vector. Training often involves solving a least squares problem or a regularized variant such as ridge regression:
其中 $\mathbf{w}$ 是权重向量。训练通常涉及求解最小二乘问题或正则化变体，例如岭回归：

$\min_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|^2.$

This is solved efficiently using matrix factorizations.
使用矩阵分解可以有效地解决这个问题。

### Singular Value Decomposition (SVD)
奇异值分解（SVD）

The SVD of a matrix $X$ is
矩阵 $X$ 的 SVD 为

$X = U \Sigma V^T,$

where $U, V$ are orthogonal and $\Sigma$ is diagonal with nonnegative entries (singular values).
其中 $U, V$ 是正交的， $\Sigma$ 是对角的，具有非负项（奇异值）。

*   Singular values measure the importance of directions in feature space.
    奇异值衡量特征空间中方向的重要性。
*   SVD is used for dimensionality reduction (low-rank approximations), topic modeling, and recommender systems.
    SVD 用于降维（低秩近似）、主题建模和推荐系统。

### Eigenvalues in Machine Learning
机器学习中的特征值

*   PCA (Principal Component Analysis): diagonalization of the covariance matrix identifies directions of maximal variance.
    PCA（主成分分析）：协方差矩阵的对角化确定了最大方差的方向。
*   Spectral clustering: uses eigenvectors of the Laplacian to group data points into clusters.
    谱聚类：使用拉普拉斯算子的特征向量将数据点分组成聚类。
*   Stability analysis: eigenvalues of Hessian matrices determine whether optimization converges to a minimum.
    稳定性分析：Hessian 矩阵的特征值决定优化是否收敛到最小值。

### Neural Networks
神经网络

Even deep learning, though nonlinear, uses linear algebra at its core:
即使是深度学习，尽管是非线性的，其核心也使用线性代数：

*   Each layer is a matrix multiplication followed by a nonlinear activation.
    每一层都是矩阵乘法，然后是非线性激活。
*   Training requires computing gradients, which are expressed in terms of matrix calculus.
    训练需要计算梯度，以矩阵微积分来表示。
*   Backpropagation is essentially repeated applications of the chain rule with linear algebra.
    反向传播本质上是链式法则与线性代数的重复应用。

### Why this matters
为什么这很重要

Machine learning models often involve datasets with millions of features and parameters. Linear algebra provides the algorithms and abstractions that make training and inference possible. Without it, large-scale computation in AI would be intractable.
机器学习模型通常涉及具有数百万个特征和参数的数据集。线性代数提供了使训练和推理成为可能的算法和抽象。如果没有它，人工智能中的大规模计算将变得难以处理。

### Exercises 10.4
练习10.4

1.  Show that ridge regression leads to the normal equations
    证明岭回归可以得出正态方程

$$
(X^T X + \lambda I)\mathbf{w} = X^T \mathbf{y}.
$$

2.  Explain how SVD can be used to compress an image represented as a matrix of pixel intensities.
    解释如何使用 SVD 来压缩以像素强度矩阵表示的图像。
    
3.  For a covariance matrix $\Sigma$, show why its eigenvalues represent variances along principal components.
    对于协方差矩阵 $\Sigma$ ，说明为什么它的特征值表示沿主成分的方差。
    
4.  Give an example of how eigenvectors of the Laplacian matrix can be used for clustering a small graph.
    举例说明如何使用拉普拉斯矩阵的特征向量对小图进行聚类。
    
5.  In a neural network with one hidden layer, write the forward pass in matrix form.
    在具有一个隐藏层的神经网络中，以矩阵形式写出前向传递。