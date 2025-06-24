# 环境安装

    git clone https://github.com/grammarly/gector.git
    cd gector

    mkdir -p models/roberta
    # 版本太旧，不能下载了
    wget https://grammarly-nlp-data.s3.amazonaws.com/gector/models/roberta_1_gector.th -O models/roberta/roberta_1_gector.th
    # 最新
    wget https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th -O models/roberta/roberta_1_gectorv2.th

    curl https://pyenv.run | bash

    pyenv install 3.7.10
    pyenv virtualenv 3.7.10 gector 
    pyenv activate gector 

    pip install --upgrade pip setuptools wheel
    pip install jsonnet==0.17.0
    pip install -r requirements.txt --prefer-binary

# Grammarly 主要有哪些产品

Grammarly 的产品体系围绕**语言智能辅助写作**构建，主要分为以下几大类：

---

### 🔤 1. **Grammarly Free（免费版）**

提供基本写作建议，包括：

* 拼写检查
* 基础语法和标点纠错
* 字词清晰度建议

适合日常邮件、社交媒体或学习场景。

---

### 💼 2. **Grammarly Premium（付费版，个人用户）**

在免费版基础上增强功能，包括：

* **高级语法与拼写检查**
* **语调调整**（Tone detection & suggestions）
* **词汇增强建议**（词语多样性）
* **简洁性改进**
* **重写建议**（Rewrite Suggestions）
* **抄袭检测**（Plagiarism detection）
* AI 生成内容建议（如自动续写、改写）

---

### 👥 3. **Grammarly Business（企业版）**

为团队和组织设计，提供：

* 所有 Premium 功能
* **品牌语调指南**（Brand tones）
* **风格指南与团队规范**
* **分析仪表盘**（团队写作数据分析）
* **用户管理与SSO登录支持**
* 与 Slack、Google Docs、Jira 等常用工具集成

适用于内容团队、客户支持、市场部门等。

---

### 🧠 4. **GrammarlyGO（生成式AI产品）**

Grammarly 的 **生成式写作AI助手**，具备：

* 自动起草邮件、博客、回复等
* 基于上下文生成个性化文案
* 自动改写、简化、扩展文本
* 用户可选择语气、受众、格式等参数

这是 Grammarly 应对 ChatGPT 类竞争的重要产品。

---

### 🧩 5. **浏览器扩展/插件**

支持以下平台直接写作增强：

* Chrome、Firefox、Safari、Edge 插件
* Google Docs 插件
* Gmail、LinkedIn、Twitter 等网页输入框实时纠错

---

### 📱 6. **Grammarly Keyboard（移动端键盘）**

适用于 iOS 和 Android：

* 替代原生输入法
* 提供实时写作建议
* 支持语调分析和词汇推荐

---

### 🔗 7. **Grammarly for Education（教育版）**

专为学校、大学、图书馆提供：

* 写作支持
* 抄袭检测
* 学术风格建议
* 教师仪表盘和学生进度跟踪

---

### 🧑 8. **Grammarly for Developers**

开放 API 和 SDK，允许开发者将 Grammarly 功能集成到自家应用中。

* Grammarly Text Editor SDK
* 提供编辑器增强能力（拼写、语法、风格建议）

# 有哪些可以在终端推理的小模型，适合做英语语法修正的？


## ✅ 主流轻量语法修正模型（可本地终端运行）

### 1. **GECToR（Grammatical Error Correction: Tag, Not Rewrite）**

* 📌 由 Yandex Research 发布
* ✅ **支持本地运行（PyTorch）**
* 💡 使用标签化序列标注方式（高效而快速）
* ⚙️ 支持多种模型变体（RoBERTa、BERT-base）
* 🚀 轻量模型推荐：

  * `gector/roberta-base`（适合本地 CPU 或小 GPU）

### 2. **T5-small / T5-base 微调版（Grammar Correction）**

* 📌 来自 Google 的 T5 模型，可微调实现语法纠正
* 🤖 Hugging Face 上有多个轻量版本，如：

  * `prithivida/grammar_error_correcter_v1`
* ✅ 支持终端 + HuggingFace Transformers + CPU/GPU 本地推理
* ⚠️ `t5-small` 版本内存需求低，可运行在中端 PC 上

---

### 3. **nanoGPT + 语法纠错微调模型（自定义）**

* 💡 若你想做极简部署，可使用 GPT2 或 nanoGPT 自行微调模型，用于特定语法任务
* 🧠 优点：完全离线，训练+推理自控
* ⚠️ 缺点：需自己准备 GEC 数据（如 CoNLL-2014）

---

## 🧰 其他工具型模型（或平台可导出模型）

| 工具名                              | 特点                        | 可否终端运行         |
| -------------------------------- | ------------------------- | -------------- |
| **LanguageTool (Open-source)**   | 规则+ML混合，支持 CLI 模式         | ✅              |
| **Gramformer (基于BART)**          | HuggingFace 上模型包装器        | ✅（但模型略大）       |
| **LLaMA 2 7B / Mistral 微调版**     | 通用 + grammar 修正 fine-tune | ⚠️ 不推荐终端无GPU运行 |
| **spaCy + Transformer pipeline** | 可加载 GEC transformer 模块    | ✅              |

---

## 🔍 推荐组合（轻量 + 易用）

* ✅ **GECToR + roberta-base**：最轻巧 + 推理速度快
* ✅ **T5-small Grammar corrector**：通用性强，部署简便
* ✅ **LanguageTool CLI**：完全无模型推理（基于规则），低资源场景理想选择

# GecBERTModel 介绍

你使用的 `GecBERTModel` 是 [**GECToR**](https://github.com/grammarly/gector) 模型的核心类，它是由 Grammarly 开源的一个 **基于 Transformer 的语法纠错系统**。它的设计理念是：将语法纠错任务建模为“编辑操作预测”，即对原句的每个词预测“保留 / 替换 / 删除 / 插入”。

---

## ✅ 这个 model 可以做什么？

**GecBERTModel 可以用于：**

| 能力          | 说明                           |
| ----------- | ---------------------------- |
| ✅ 英语语法纠错    | 检查并纠正语法错误，如主谓不一致、时态错误、拼写、介词等 |
| ✅ 批量文本纠正    | 支持批量输入句子                     |
| ✅ 多轮纠错迭代    | 可通过多次迭代逐步修复复杂错误              |
| ✅ 支持多种预训练模型 | 如 RoBERTa、BERT、XLNet 等       |
| ✅ 可训练       | 支持自定义训练自己的语法纠错模型             |

---

## 🧩 模型构造函数参数详解

```python
GecBERTModel(
    vocab_path='data/output_vocabulary',
    model_paths=["models/roberta/roberta_1_gectorv2.th"],
    max_len=50,
    min_len=3,
    iterations=3,
    lowercase_tokens=True
)
```

### 🔹 1. `vocab_path`

* 类型：`str`
* 含义：GECToR 模型使用的输出词汇表（即它能预测的编辑操作）
* 作用：用于 mapping 动作标签（如 `$KEEP`, `$DELETE`, `change:go→goes` 等）
* 默认：用项目自带的 `output_vocabulary` 即可

---

### 🔹 2. `model_paths`

* 类型：`List[str]`
* 含义：模型权重文件路径，通常是一个 `.th` 文件（AllenNLP 保存的 pytorch state dict）
* 作用：你可以加载多个模型做 ensemble（集成预测）
* 示例：`roberta_1_gectorv2.th` 是一个基于 RoBERTa-base 训练的模型

---

### 🔹 3. `max_len`

* 类型：`int`
* 含义：模型处理句子的最大 token 长度（超过会截断）
* 作用：RoBERTa 的上限是 512，但实际中设 50\~128 比较快也够用
* 影响：太短可能截断重要内容，太长会降低速度

---

### 🔹 4. `min_len`

* 类型：`int`
* 含义：句子最小长度，小于此长度的句子将跳过处理
* 作用：避免对无意义短句（如“Hi”）浪费计算资源

---

### 🔹 5. `iterations`

* 类型：`int`
* 含义：纠错迭代次数
* 作用：GECToR 是“增量修正”模型，需要多轮纠正复杂句子错误
* 推荐值：2\~3 轮最常见；太多可能过拟合或耗时

---

### 🔹 6. `lowercase_tokens`

* 类型：`bool`
* 含义：是否将输入 tokens 转小写再处理
* 作用：部分模型训练时用小写文本（如 `bert-base-uncased`），需统一大小写
* 注意：如果你模型是大小写敏感的（如 `roberta-base`），应设为 `False`（不过 `True` 也可工作）

---

## 🧠 模型原理简述

GECToR 的思想是：

> 将语法纠错建模为序列标注任务 —— 对每个词预测一个操作标签（如 `$KEEP`, `change:is→are` 等），而不是直接生成整个句子。

优点：

* 推理快（只预测 token 标签）
* 可以 fine-tune 预训练模型（如 RoBERTa）
* 多轮叠加修复复杂语法错误

---

## 📌 使用 GecBERTModel 的常见方法

| 方法                                  | 用途                        |
| ----------------------------------- | ------------------------- |
| `handle_batch(sentences)`           | 对一批句子进行纠错，返回（纠正后句子, 总改动数） |
| `predict_for_sent(sentence_tokens)` | 单句预测，返回每个词的编辑操作           |
| `predict(sentences)`                | 对输入句子纠正，但不返回 token 级标注    |
| `preprocess(sentences)`             | 对句子进行 tokenize、mask 等预处理  |

---

## ✅ 适用场景

* 教育类应用（如作文纠错、英语学习工具）
* 英语 NLP 预处理增强（如 grammar-aware translation）
* 英文邮件、文章写作辅助（像 Grammarly）
