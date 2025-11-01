# 大模型训练

- 什么是 LLM？它的底层原理是什么？
- LLM 有哪些分类？
- 什么是 LLM 的训练？
- 什么是 LLM 的推理？
- 什么时候应该训练自己的大模型？
- 如何训练自己的大模型？
- 什么时候应该用 RAG？
- hugging face 有哪些帮助训练大模型的库？
- 训练时如何降低内存和显卡使用量？
- 大模型压缩有哪些技术？

## 什么是 LLM，他们有哪些分类？

**LLM（大语言模型）** 是一种基于深度学习的模型，训练目标是**理解和生成自然语言文本**。
它通过在大规模文本数据上训练，学习语言的统计规律、语义结构、推理关系等。

LLM 是一个 **自监督学习的概率模型**，目标是学习语言的分布：
也就是：**给定前面的词，预测下一个最可能出现的词。**

通过大量语料训练，模型学会了：

* 词与词的语义关系
* 上下文依赖
* 推理逻辑与世界知识

Transformer 是LLM 的心脏，通常分为两部分：

| 模块          | 作用            | 常见用途       |
| ----------- | ------------- | ---------- |
| **Encoder** | 编码输入信息，提取语义特征 | BERT 等理解模型 |
| **Decoder** | 根据上下文逐词生成输出   | GPT 等生成模型  |


一个 Transformer 由多个层（block）组成，每个层由以下模块组成：

```
输入嵌入 → 多头自注意力 → 残差 + LayerNorm → 前馈网络 (Feed Forward) → 残差 + LayerNorm → 输出
```
- 输入嵌入 (Input Embedding): 把每个词或 token 映射成向量，加上 位置编码 (Positional Encoding)，让模型知道词序。
- 自注意力机制（Self-Attention）：让模型能“聚焦”在与当前词相关的信息上
- 多头注意力（Multi-Head Attention）：学习不同类型的关系（语法、语义、上下文依赖等）。
- 前馈网络（Feed Forward Network, FFN）：增加模型的非线性和表达能力
- 残差连接 + Layer Normalization：让梯度更稳定，训练更容易收敛。


其中自注意力机制（Self-Attention）是 Transformer 的灵魂。每个词在计算自己的表示时，会“关注”其他词的重要性。

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


## LLM 有哪些分类？

按训练目标分类

| 类型                              | 代表模型              | 训练任务                            | 特点                     |
| ------------------------------- | ----------------- | ------------------------------- | ---------------------- |
| **因果语言模型（Causal LM）**           | GPT、LLaMA、Mistral | 预测下一个词（Next Token Prediction）   | 擅长**文本生成**（写作、对话、编程）   |
| **掩码语言模型（Masked LM）**           | BERT、RoBERTa      | 预测被遮盖的词（Masked Word Prediction） | 擅长**理解任务**（分类、抽取、推理）   |
| **双向 + 自回归模型（Encoder–Decoder）** | T5、BART           | 结合掩码与生成                         | 擅长**生成+理解**混合任务（翻译、摘要） |


 按架构结构分类

| 架构类型                | 结构           | 代表模型    | 优点      |
| ------------------- | ------------ | ------- | ------- |
| **Encoder-only**    | 只用编码器（双向注意力） | BERT    | 强理解能力   |
| **Decoder-only**    | 只用解码器（单向注意力） | GPT 系列  | 强生成能力   |
| **Encoder–Decoder** | 编码器+解码器      | T5、BART | 平衡理解与生成 |


总结

| 维度  | 掩码语言模型（Masked LM） | 因果语言模型（Causal LM） |
| --- | ----------------- | ----------------- |
| 上下文 | 双向                | 单向                |
| 用途  | 理解                | 生成                |
| 架构  | Encoder-only      | Decoder-only      |
| 举例  | BERT、RoBERTa      | GPT、LLaMA、Mistral |

## 什么是 LLM 的训练？

**训练（Training）** 就是让模型通过大量数据“学习”语言规律与人类意图的过程。从最顶层看，大语言模型的训练通常分为四个阶段：

- 预训练（Pretraining）：让模型学习语言的统计规律、语法结构、语义关联、世界知识。
    - 模型能“理解语言”，但不一定懂“人类意图”。
- 监督微调（Supervised Fine-Tuning, SFT）：让模型学会“按照人类指令”去做事（Instruction Following）。
    -  模型变得更“听话”，会按任务类型回答，但可能仍有偏见或不安全行为。
- 对齐阶段（Alignment）：让 LLM 从“会写”进化到“会沟通”的关键阶段。
    - 模型更安全、礼貌、有逻辑，减少幻觉、歧视、有害输出    
-  持续微调 / 增量训练（Continual Finetuning）：让模型适应新数据、新知识、新领域。

| 阶段   | 数据来源  | 学习方式       | 目标     | 输出能力   |
| ---- | ----- | ---------- | ------ | ------ |
| 预训练  | 大规模文本 | 自监督        | 学语言、知识 | 通用语言能力 |
| 监督微调 | 指令数据  | 有监督        | 学任务    | 指令遵循   |
| 对齐   | 偏好数据  | 强化学习 / DPO | 学价值观   | 符合人类预期 |
| 增量微调 | 特定语料  | 有监督或少样本    | 学领域知识  | 专业任务表现 |


# other
pip install huggingface_hub
HF_ENDPOINT=https://hf-mirror.com hf download bigscience/mt0-large --local-dir ./mt0-large

export HF_ENDPOINT=https://hf-mirror.com
pip install -q peft transformers datasets

Transformers 快速入门
https://huggingface.co/docs/transformers/quicktour

peft notebooks
https://huggingface.co/spaces/PEFT/causal-language-modeling

因果模型（Causal language modeling）微调教程
https://huggingface.co/docs/transformers/tasks/language_modeling

因果模型（Causal language modeling）微调 notebooks
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb

微调相关：
- Fine-Tuning Large Language Models with LoRA: A Practical Guide https://www.christianmenz.ch/programming/fine-tuning-large-language-models-with-lora-a-practical-guide/?utm_source=chatgpt.com
- In-depth guide to fine-tuning LLMs with LoRA and QLoRA https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora?utm_source=chatgpt.com
- Efficient Fine-tuning with PEFT and LoRA https://heidloff.net/article/efficient-fine-tuning-lora/?utm_source=chatgpt.com 
    - https://www.philschmid.de/fine-tune-flan-t5-peft
    - https://huggingface.co/blog/4bit-transformers-bitsandbytes
- Fine-Tuning Large Language Models (LLMs) https://towardsdatascience.com/fine-tuning-large-language-models-llms-23473d763b91/
- Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA https://huggingface.co/blog/4bit-transformers-bitsandbytes

摘录

- 使用 RAG 时，我们不会对模型进行任何操作，它就是原始模型，数据只是随请求一起传递。而微调则不同，它使用额外的领域特定数据重新训练模型，因此我们会更改模型权重，最终得到一个包含领域特定知识的新模型。
- 你可以创建两个低秩矩阵，然后将它们相乘得到一个大矩阵。
- 微调后的模型可以通过提供定制化功能，从多方面提升业务绩效。微调模型有助于您根据自身需求和知识进行定制。
- 您可以使用 RAG 管道来定制模型，但有时知识量过于庞大，仅靠嵌入和相似性搜索是不够的，这时就需要通过微调进行定制。
- 微调不仅可以提高基础模型的性能，而且较小的（微调后的）模型通常在其训练的任务集上优于较大的（更昂贵的）模型 
- 一个主要问题是，LLM 的上下文窗口有限。因此，对于需要庞大知识库或领域特定信息的任务，模型的性能可能欠佳[1]。微调模型可以通过在微调过程中“学习”这些信息来避免这个问题。
- 微调模型通常有三种方法 ：自监督学习、监督学习和强化学习。