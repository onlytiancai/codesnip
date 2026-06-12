# 神经网络小课堂 · 互动教程

> **面试 AI 岗第一题——手写 MLP，从零讲给小学生听**。
> 纯前端单页应用，0 构建、0 npm；中英双语；明暗两色主题；进度 / 答题 / 证书一应俱全。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Vue 3.5](https://img.shields.io/badge/Vue-3.5-4FC08D)](https://vuejs.org)
[![Chart.js 4.4](https://img.shields.io/badge/Chart.js-4.4-FF6384)](https://www.chartjs.org)
[![KaTeX 0.16](https://img.shields.io/badge/KaTeX-0.16-329796)](https://katex.org)

---

## ✨ 特色

- 🎓 **10 章系统课程**：从数学预备（函数 / 坐标系 / 斜率 / 向量 / 矩阵 / 链式法则）到亲手手写 MLP 解决 XOR
- 🧪 **32+ 道测试题**：单选 / 多选 / 简答，每题带"查看答案解析"
- 🏆 **结业证书**：完成 10 章 + 60% 正确率即可下载 PNG 证书
- 🌍 **中英双语**：右上角一键切换
- 🎨 **明暗双主题**：右上角切换
- 💾 **进度持久化**：localStorage 记录每章进度、每题答案、主题 / 语言 / 证书
- 🎬 **互动数学组件**：sigmoid 滑块、直线斜率、训练曲线、计算图
- 📜 **Vue 组件 + 类 MDX 语法**：用 `::: quiz / chart / graph / network / train-demo / formula ... :::` 容器在 markdown 中嵌入 Vue 组件
- 🪶 **极轻量**：单 HTML + JSON + markdown，无构建，无 npm

---

## 🚀 快速开始

需要 Python 3（任何版本）做静态服务器，**没有其他依赖**。

```bash
# 1. 进入项目
cd /Users/huhao/src/codesnip/python/ai/009

# 2. 起静态服务器
/Users/huhao/.pyenv/versions/3.11.9/bin/python3 -m http.server 8765
# 或者：python3 -m http.server 8765

# 3. 浏览器打开
open http://localhost:8765
```

> ⚠️ **不能用 `file://` 打开**——浏览器会拒绝 `fetch('chapters.json')`。
> 必须通过 HTTP 服务器（即使是本地的）访问。

### 重新生成配图

如果修改了某个章节的图，重新生成：

```bash
cd /Users/huhao/src/codesnip/python/ai/009
bash scripts/gen_all.sh
# 或单独生成一章
/Users/huhao/.pyenv/versions/3.11.9/bin/python3 scripts/gen_ch03.py
```

依赖：`numpy` + `matplotlib`（参见 `scripts/requirements.txt`）。

---

## 📁 目录结构

```
009/
├── index.html                          # 唯一 HTML 入口（SPA 容器）
├── chapters.json                       # 章节元数据
├── progress.config.js                  # 进度阈值常量
├── README.md                           # 本文件
│
├── content/                            # 章节 markdown（按 ch## 编号）
│   ├── ch00_math_{zh,en}.md            # 第 0 章：数学准备
│   ├── ch01_intro_{zh,en}.md
│   ├── ch02_neuron_{zh,en}.md
│   ├── ch03_perceptron_{zh,en}.md
│   ├── ch04_xor_{zh,en}.md
│   ├── ch05_mlp_{zh,en}.md
│   ├── ch06_forward_{zh,en}.md
│   ├── ch07_loss_{zh,en}.md
│   ├── ch08_backprop_{zh,en}.md
│   ├── ch09_gradient_{zh,en}.md
│   └── ch10_train_{zh,en}.md
│
├── assets/
│   ├── css/
│   │   ├── theme.css                   # CSS 变量 + 主题切换
│   │   ├── app.css                     # 布局 / 排版 / 头部 / 侧边栏
│   │   ├── quiz.css                    # 测试题样式
│   │   └── cert.css                    # 证书样式 + @media print
│   ├── js/
│   │   ├── app.js                      # Vue 入口（createApp + 挂载）
│   │   ├── router.js                   # vue-router hash 路由
│   │   ├── store.js                    # 迷你 store（reactive + provide/inject）
│   │   ├── progress.js                 # localStorage 读写 + schema 兼容
│   │   ├── markdown.js                 # marked 配置 + ::: 容器扩展
│   │   ├── i18n.js                     # UI 翻译表
│   │   └── components/                 # 13 个 Vue 组件
│   │       ├── AppShell.js · Sidebar.js · HomeView.js · ChapterView.js · CertView.js
│   │       ├── Quiz.js · ChartBlock.js · ComputeGraph.js
│   │       ├── NetworkViz.js · TrainDemo.js · Formula.js
│   │       └── SigmoidSlider.js · MathSlider.js
│   ├── images/                         # 21 张配图（matplotlib 生成）
│   ├── favicon.svg
│   └── cert-bg.svg                     # 证书底图
│
└── scripts/                            # 配图生成脚本
    ├── _fonts.py                       # CJK 字体配置（复用 000.md 速查）
    ├── gen_ch00_math.py                # 6 张数学配图
    ├── gen_ch01.py ~ gen_ch10.py       # 主线配图
    ├── gen_all.sh                      # 一键跑全部
    └── requirements.txt                # numpy, matplotlib
```

---

## 📚 章节内容

| # | 标题 | 内容 |
|---|---|---|
| 0 | 数学准备：6 件兵器 | 函数 / 坐标系 / 斜率 / 向量 / 矩阵 / 链式法则（独立预备章） |
| 1 | 什么是神经网络？ | 决策小精灵的故事（0 数学） |
| 2 | 一个神经元 = 一个函数 | 公式 y = σ(w·x + b) 拆解 |
| 3 | 感知机：最简单的学习机 | step + 错了就改 |
| 4 | XOR 难题 | 一条直线分不开 |
| 5 | MLP：把神经元叠成网络 | 2 → 4 → 1 + 两条直线组合 = 弯线 |
| 6 | 前向传播 | X·W+b → σ → a·W+b → σ = ŷ + 手算 |
| 7 | 损失函数 | MAE / MSE / BCE 哪个好 |
| 8 | 反向传播 | 链式法则 + 4 步公式 + 手算 |
| 9 | 梯度下降 + 训练循环 | 4 步循环 + 学习率 |
| 10 | 手写 MLP 解决 XOR + 证书 | 端到端 numpy 代码 + 颁发证书 |

**总计**：10 章（11 个 markdown 文件 / 语言） + 32 道测试题 + 21 张配图 + 1 份证书

---

## ✍️ 如何加新章

**示例**：新增第 11 章《卷积神经网络简介》。

### 1. 在 `content/` 下创建 2 个 markdown 文件

`content/ch11_cnn_zh.md`：
```markdown
# 第 11 章 卷积神经网络简介

## 11.1 为什么需要 CNN

> ...

::: quiz q11-1 single
CNN 主要处理什么类型的数据？
- A: 表格数据
- B: 图像
- C: 文本
- D: 音频

answer: B

> CNN = Convolutional Neural Network，主要处理图像。
:::

::: chart caption="一张 28×28 的手写数字"
![手写数字](assets/images/ch11_mnist.png)
:::
```

`content/ch11_cnn_en.md`（对应英文版）。

### 2. 在 `chapters.json` 中追加新章

```json
{
  "id": "ch11",
  "order": 11,
  "title_zh": "卷积神经网络简介",
  "title_en": "Intro to Convolutional Neural Networks",
  "summary_zh": "为什么 CNN 适合处理图像",
  "summary_en": "Why CNNs are great for images",
  "file_zh": "content/ch11_cnn_zh.md",
  "file_en": "content/ch11_cnn_en.md",
  "prerequisites": ["ch05"],
  "image_count": 1,
  "image_prefix": "ch11_cnn",
  "has_quiz": true,
  "quiz_count": 3,
  "estimated_minutes": 30
}
```

### 3. 在 `scripts/` 下创建配图脚本（可选）

```python
# scripts/gen_ch11.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _fonts import new_figure, save

def mnist_image():
    # ... 画一张 28x28 的 0/1 灰度图
    pass

if __name__ == "__main__":
    mnist_image()
```

```bash
/Users/huhao/.pyenv/versions/3.11.9/bin/python3 scripts/gen_ch11.py
```

### 4. 重新加载浏览器即可

新章节会出现在侧边栏和首页卡片中。

---

## 🛠️ 测试题特殊 markdown 语法

### 单选题

```markdown
::: quiz q3-1 single
感知机公式 `step(w·x + b)` 中 `step(z)` 是什么？

- A: 求和
- B: 大于等于 0 输出 1，否则 0
- C: 取最大值
- D: 随机

answer: B

> step 是阶跃函数：`z ≥ 0` 输出 1，`z < 0` 输出 0。感知机用它做"二分类"。
:::
```

### 多选题

```markdown
::: quiz q1-2 multiple
神经元和大脑神经细胞的相同点是？（多选）

- A: 都有输入信号
- B: 都能学习
- C: 结构一模一样
- D: 都用电信号

answer: A,B
:::
```

### 简答题

```markdown
::: quiz q1-3 short placeholder="用 1 句话告诉爸妈什么是神经网络"
请用 1 句话向你爸妈解释"什么是神经网络"。

> 参考答案：神经网络是一堆数学公式，能从数据里"学"出规律。比如见过 1000 张猫的照片后，能认出新照片里有没有猫。
:::
```

### 其他容器

| 容器 | 用途 | 示例 |
|---|---|---|
| `::: chart caption="..."` | 静态配图 | `![描述](assets/images/xxx.png)` |
| `::: graph` | 计算图（SVG） | 神经网络 / 公式树 |
| `::: network` | 可交互网络拓扑 | MLP 节点可悬停高亮 |
| `::: train-demo :steps=200 :lr=0.5` | 训练演示（chart.js） | 损失曲线 + 准确率 |
| `::: formula` | 独立公式（KaTeX） | 大块公式展示 |

---

## 🧰 技术栈

| 库 | 版本 | 来源 | 用途 |
|---|---|---|---|
| Vue 3 | 3.5.13 | jsdelivr | 反应式 UI 框架 |
| Vue Router 4 | 4.4.5 | jsdelivr | hash 路由 |
| marked | 14.1.3 | jsdelivr | markdown → HTML |
| DOMPurify | 3.2.4 | jsdelivr | XSS 清洗 |
| html2canvas | 1.4.1 | jsdelivr | 证书 → PNG 下载 |
| chart.js | 4.4.1 | cdn.ihuhao.com | 训练曲线 |
| KaTeX | 0.16.9 | cdn.ihuhao.com | 数学公式 |
| numpy | ≥1.24 | PyPI | 配图脚本 |
| matplotlib | ≥3.7 | PyPI | 配图脚本 |

**跳过 Pinia**（无 IIFE 构建）→ 用 `reactive()` + `provide/inject` 写迷你 store
**跳过 markdown-it-container**（无浏览器打包）→ 用 marked 自定义扩展

---

## 📐 配色（清亮淡雅）

| 角色 | Light | Dark |
|---|---|---|
| 背景 | `#fafbff` | `#0f1419` |
| 卡片 | `rgba(255,255,255,0.95)` | `rgba(22,27,34,0.92)` |
| 文字 | `#1e293b` | `#e6edf3` |
| 主色（薄荷绿） | `#10b981` | `#34d399` |
| 副色（天蓝） | `#0ea5e9` | `#38bdf8` |
| 警告 | `#f59e0b` | `#fbbf24` |
| 危险 | `#ef4444` | `#f87171` |

圆角统一 12px；字体 `-apple-system, "PingFang SC", "Microsoft YaHei", sans-serif`。

---

## 🔧 常见问题

### Q: 打开页面显示"😢 启动失败"？
A: 检查：
1. 是否通过 HTTP 服务器访问（不能 `file://`）
2. 浏览器控制台是否有 404 错误
3. `chapters.json` 是否在项目根目录

### Q: 切换英文不生效？
A: 检查 `content/` 下是否有对应的 `_en.md` 文件。文件名格式：`ch##_xxx_en.md`

### Q: 答过的题"答对/答错"不显示？
A: 检查 localStorage 是否被禁用（隐私模式 / 浏览器设置）

### Q: 证书无法下载？
A: 浏览器可能拦截了弹窗/下载。允许本站点的下载权限。

### Q: 配图脚本跑失败？
A: 大概率是字体问题。`scripts/_fonts.py` 已处理 CJK 字体回退（macOS → Windows → Linux）。如果仍报字体警告，不影响 PNG 生成。

---

## 📜 License

MIT

---

## 🙏 致谢

- [Vue 3](https://vuejs.org)、[vue-router](https://router.vuejs.org)、[marked](https://marked.js.org)、[DOMPurify](https://github.com/cure53/DOMPurify)、[html2canvas](https://html2canvas.hertzen.com)、[Chart.js](https://www.chartjs.org)、[KaTeX](https://katex.org)
- [006_01.html](../006_01.html) — 主题切换 / KaTeX 注入模板
- [000.md](../000.md) — CJK 字体配置速查
