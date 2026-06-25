::: en
# How LLMs Actually Work
:::

::: zh
# LLMs 实际工作原理
:::

::: en
> **Source** ：[0xkato.xyz](https://www.0xkato.xyz/how-llms-actually-work/)  
> **Published** ：June 01, 2025 - 26 mins
:::

::: zh
> **原文出处** ：[0xkato.xyz](https://www.0xkato.xyz/how-llms-actually-work/)  
> **发表日期** ：2025年6月1日 - 26分钟阅读  
> **译者注** ：本文为大型语言模型（LLM）内部机制的双语对照版本，原文为英文，中文部分由 AI 翻译并人工校对。文中所有图片保留原链接并附中文要点说明。专有名词首次出现时附原文。
:::

---

::: en
## Tokenization
:::

::: zh
## 分词
:::

::: en
Models don't read text directly. They read integer IDs. The step that converts your prompt into a sequence of those integers.

That conversion step is called tokenization. A tokenizer takes a string and produces a sequence of integers, where each integer points to an entry in a fixed vocabulary. Modern LLM vocabularies usually contain tens of thousands to a few hundred thousand entries.
:::

::: zh
模型并不直接读取文本，而是读取整数 ID。 **分词（Tokenization）** 是将你的输入文本转换为整数序列的步骤。

分词器（Tokenizer）接收一段字符串，输出一个整数序列，每个整数对应固定词表（Vocabulary）中的一个条目。现代 LLM 的词表通常包含几万到几十万不等的条目。
:::

::: en
> **Tiny explainer: token ID**  
> A token ID is the integer the model uses for one vocabulary entry. The model works with the number, not the written word itself.
:::
::: zh
> **简明说明: 词元 ID**  
> 词元 ID 是模型用于标识词表条目的整数。模型操作的是数字，而非文字本身。
:::

::: en
Tokens aren't usually whole words. They're usually subword pieces. The word "tokenization" might split into ["token", "ization"]. The word "running" might split into ["run", "ning"]. The reason is efficiency. Whole-word vocabularies are too big and don't generalize to new words. Character-level vocabularies are too small and force the model to learn even the simplest patterns from scratch. Subword tokenization sits in the middle.
:::

::: zh
Token 通常不是完整的单词，而是 **子词（subword）** 片段。比如 "tokenization" 可能被拆分为 ["token", "ization"]，"running" 可能被拆分为 ["run", "ning"]。这样做是为了效率：全词词表太大，且无法泛化到新词；字符级词表太小，模型需要从零学习最简单的模式。子词分词法处于两者之间。
:::

::: en
> **Tiny explainer: vocabulary**  
> The vocabulary is the tokenizer's fixed list of pieces. Each piece has an ID, and the model can only directly receive IDs from that list.
:::
::: zh
> **简明说明: 词表**  
> 词表是分词器的固定条目列表，每个条目对应一个 ID，模型只能直接接收来自该列表的 ID。
:::

::: en
The trade-off shows up in places people don't expect. The classic example: ask an LLM how many R's are in "strawberry." LLMs used to get it wrong. That's not the model failing at counting. It's the model not operating on letters directly, only token IDs that happen to spell out a word a human would split letter by letter.
:::

::: zh
这种权衡出现在一些意想不到的地方。经典例子：问 LLM "strawberry" 中有多少个 R，模型常常答错。这不是模型不会数数，而是因为模型并非直接操作字母，它只处理 token ID——这些 ID 恰好对应人类会逐字母拆分的单词。
:::

![](https://0xkato.xyz/assets/transformer-tokenization.png)

::: en
Different model families use different tokenizers. GPT models use Byte Pair Encoding variants. SentencePiece is common in LLaMA-style models. The choice matters for compute (fewer tokens means less work) and for things like multilingual coverage, but the basic shape is the same. Text in, integers out.

Now that the prompt is a sequence of integers, the next step is to give those integers meaning.
:::

::: zh
不同的模型家族使用不同的分词器。GPT 系列使用字节对编码（Byte Pair Encoding）的变体；SentencePiece 在 LLaMA 风格模型中较为常见。分词器的选择会影响计算量（更少的 token 意味着更少的工作量）以及多语言覆盖率等，但基本形式相同： **文本输入，整数输出** 。

现在输入已经变成整数序列，下一步是赋予这些整数以意义。
:::

---

::: en
## Embeddings
:::

::: zh
## 嵌入
:::

::: en
A token ID like `1024` is just a row index. It doesn't mean anything by itself. The thing that gives it meaning is a giant table called the embedding matrix.

Every model has one. It has one row per entry in the vocabulary, and each row is a long vector of numbers. The length of each row is the model's hidden size. In many 7B-class models, that means 4,096 numbers per token. Larger models usually use wider vectors.
:::

::: zh
像 `1024` 这样的 token ID 只是一个行索引，本身没有任何含义。赋予它意义的是一个叫做 **嵌入矩阵（Embedding Matrix）** 的巨大表格。

每个模型都有一个嵌入矩阵。矩阵的每一行对应词表中的一个条目，每一行都是一个很长的数字向量。每一行的长度就是模型的 **隐藏层维度（Hidden Size）** 。在许多 7B 参数级别的模型中，这意味着每个 token 对应 4,096 个数字。更大的模型通常使用更宽的向量。
:::

::: en
> **Tiny explainer: vector**  
> A vector is a list of numbers. In a transformer, each token becomes a vector so the model can do math with it.
:::
::: zh
> **简明说明: 向量**  
> 向量是一列数字。在 Transformer 中，每个 token 变成一个向量，以便模型进行数学运算。
:::

::: en
When the tokenizer hands the model an integer, the model looks up that row and uses the vector instead. That vector is the token's embedding. It's the model's representation of what that token "means," learned during training.
:::

::: zh
当分词器将一个整数传递给模型时，模型查找对应行并使用该向量。这个向量就是 token 的 **嵌入（Embedding）** ——它是模型对该 token "含义"的表示，是在训练过程中学到的。
:::

::: en
> **Tiny explainer: embedding matrix**  
> The embedding matrix is a lookup table. Token ID in, learned vector out.
:::
::: zh
> **简明说明: 嵌入矩阵**  
> 嵌入矩阵是一个查找表。输入 token ID，输出学习到的向量。
:::

::: en
The interesting property of these embeddings is that semantically similar tokens end up with similar vectors. The vector for "king" is close in space to the vector for "queen," and the vector for "Paris" is close to "France." None of this is hard-coded. It emerges from training on enough text, and the model learns these positions because they let it predict text well.
:::

::: zh
嵌入有一个有趣的特性： **语义相似的 token 最终会有相似的向量** 。"king" 的向量在空间中接近 "queen" 的向量，"Paris" 的向量接近 "France" 的向量。这些都不是硬编码的，而是从足够的文本训练中涌现出来的。模型学习这些位置是因为它们帮助模型更好地预测文本。
:::

::: en
You can do arithmetic on embeddings and it sometimes works. The famous example is `king − man + woman ≈ queen`. The geometry of embedding space carries real semantic structure, even though nobody told the model to build it that way.
:::

::: zh
你可以在嵌入上做算术运算，而且有时居然能 work。著名例子是 `king − man + woman ≈ queen`。嵌入空间的几何结构承载着真实的语义结构，尽管没有人告诉模型要这样构建。
:::

![嵌入空间类比：语义关系示意](https://0xkato.xyz/assets/transformer-embedding-analogy.png)

::: en
Worth being clear on: at this stage every token has been replaced by its embedding, but the embedding alone says nothing about where the token sits in the sequence. The vector for "dog" is the same vector whether "dog" is the first word in your prompt or the fifth. That's a problem.

That's the gap positional encoding fills.
:::

::: zh
需要明确的是：在这个阶段，每个 token 都已经替换为其嵌入，但单独的嵌入并不能说明 token 在序列中的位置。无论 "dog" 是你 prompt 的第一个词还是第五个词，它的向量都是一样的。这是一个问题。

这就是位置编码（Positional Encoding）要解决的问题。
:::

---

::: en
## Positional Encoding
:::

::: zh
## 位置编码
:::

::: en
Plain self-attention doesn't have a built-in representation of word order. Without some positional signal, it has no direct way to know that "dog" came before "bites" instead of after it.

Word order changes meaning. So the model needs another piece. It needs a way to inject the position of each token into the math.
:::

::: zh
普通的自注意力机制并没有内置的词序表示。如果没有某种位置信号，模型就无法直接知道 "dog" 是在 "bites" 之前还是之后。

词序会改变含义。所以模型需要另一个组件——一种将每个 token 的位置注入数学运算的方式。
:::

::: en
> **Tiny explainer: positional encoding**  
> Positional encoding is how the model gets order information. It tells the model where each token sits in the sequence.
:::
::: zh
> **简明说明: 位置编码**  
> 位置编码是模型获取顺序信息的方式。它告诉模型每个 token 在序列中的位置。
:::

::: en
The original transformer paper (Vaswani et al. 2017) did this by giving each position its own pattern of numbers and adding it directly to each token's embedding before any other processing. Position 1 had one pattern, position 5 had a different pattern, position 100 had another. The patterns came from sine and cosine waves at different frequencies. Now the embedding for "dog" at position 1 was different from the embedding for "dog" at position 5, just because the position pattern added to it was different.
:::

::: zh
最初的 Transformer 论文（Vaswani et al., 2017）通过为每个位置分配独特的数字模式，并将这个模式直接加到每个 token 的嵌入上来进行处理。位置 1 有一个模式，位置 5 有另一个模式，位置 100 又有不同的模式。这些模式来自不同频率的正弦和余弦波。这样，"dog" 在位置 1 的嵌入与 "dog" 在位置 5 的嵌入就不同了——仅仅因为加到上面的位置模式不同。
:::

::: en
That worked, and sinusoidal encodings were chosen partly because they can extrapolate beyond the exact sequence lengths seen during training. But additive position schemes still had two problems that became important as models scaled up.
:::

::: zh
这种方法有效，选择正弦编码的部分原因是它能够外推到训练时未见过的序列长度。但加法位置编码方案仍然有两个问题，随着模型规模扩大而变得重要。
:::

::: en
First, the embedding had to carry both meaning and position in the same set of numbers. There's only so much you can pack in.

Second, learned absolute position embeddings in particular don't generalize cleanly. If you trained on prompts up to 2,048 tokens long, the model never saw position 5,000 during training, and the embedding for that position was not learned in the same way.
:::

::: zh
首先，嵌入需要在同一组数字中同时承载含义和位置，信息密度有限。

其次，学习的绝对位置嵌入特别不能很好地泛化。如果你在最长 2,048 个 token 的 prompt 上训练，模型从未在训练中见过位置 5,000，该位置对应的嵌入学习方式也不同。
:::

::: en
Modern models mostly use a different scheme called Rotary Position Embeddings (RoPE), introduced by Su et al. in 2021 and now used in LLaMA, Mistral, Gemma, Qwen, and most other open-weight families. The intuition: instead of adding position info to each token's vector, RoPE rotates the Query and Key vectors by an angle that depends on the token's position. A token at position 1 gets a small turn, a token at position 100 gets a bigger turn. When two tokens are later compared during attention, what matters is the difference between their Query and Key rotations, which encodes how far apart they are.
:::

::: zh
现代模型大多使用一种叫做 **旋转位置编码（RoPE, Rotary Position Embeddings）** 的方案，由 Su 等人在 2021 年提出，现已被 LLaMA、Mistral、Gemma、Qwen 以及大多数开源模型家族采用。其核心思想：RoPE 不是将位置信息加到每个 token 的向量上，而是根据 token 的位置旋转 Query 和 Key 向量。位置 1 的 token 旋转角度小，位置 100 的 token 旋转角度大。当两个 token 在注意力阶段进行比较时，重要的是它们 Query 和 Key 旋转角度的 **差值** ，这编码了它们之间的距离。
:::

::: en
> **Tiny explainer: RoPE**  
> RoPE stands for Rotary Position Embeddings. Instead of adding a position vector, it rotates Query and Key vectors so relative distance shows up during attention.
:::
::: zh
> **简明说明: RoPE**  
> RoPE 即旋转位置编码（Rotary Position Embeddings）。它不是添加位置向量，而是旋转 Query 和 Key 向量，使相对距离在注意力计算时自然显现。
:::

![旋转位置嵌入按位置旋转向量](https://0xkato.xyz/assets/transformer-rope.png)

::: en
The practical advantages are real. RoPE encodes relative position naturally (which is closer to what attention actually wants). It generalizes better to longer contexts. And it doesn't add new parameters to the model.
:::

::: zh
实际优势是真实的。RoPE 自然地编码相对位置（这更接近注意力机制真正需要的）。它能更好地泛化到更长的上下文。而且不增加新的模型参数。
:::

::: en
Even with good positional encoding, modern LLMs have a documented "lost in the middle" problem (Liu et al. 2023). They use information at the start and end of long prompts more reliably than information buried in the middle. That's why prompt engineering tips like "put important context first" or "repeat key info at the end" actually help. The model isn't using every part of your prompt equally well.
:::

::: zh
即使有良好的位置编码，现代 LLM 也有一个被记录的"中间丢失"问题（Liu et al., 2023）。模型更可靠地使用长 prompt 开头和结尾的信息，而不是埋在中间的信息。这就是为什么"把重要上下文放在前面"或"在结尾重复关键信息"这样的提示工程技巧真的有效。模型并不是同等好地使用 prompt 的每个部分。
:::

---

::: en
## Attention
:::

::: zh
## 注意力机制
:::

::: en
This is the mechanism that gave the architecture its name. Attention.

Inside every transformer layer, attention does one thing. It lets each token look at the other tokens it is allowed to see and decide which ones matter for what comes next.
:::

::: zh
这就是赋予该架构名字的机制—— **注意力机制（Attention）** 。

在每个 Transformer 层中，注意力机制做一件事：它让每个 token 查看允许它看到的其他 token，并决定哪些对接下来要做的事情重要。
:::

::: en
It does this by giving each token three roles at once. Each token gets transformed into three new vectors, called Query, Key, and Value (Q, K, V).
:::

::: zh
它通过给每个 token 同时分配三个角色来实现这一点。每个 token 被转换成三个新的向量，分别叫做 **Query（查询）、Key（键）、Value（值），简称 Q、K、V** 。
:::

::: en
> **Tiny explainer: Q, K, V**  
> Query means "what am I looking for," Key means "what do I match with," and Value is the information that gets copied when the match is strong.
:::
::: zh
> **简明说明: Q, K, V**  
> Query（查询）表示"我在寻找什么"，Key（键）表示"我与什么匹配"，Value（值）是匹配强时会被复制的信息。
:::

::: en
- The Query asks, "what am I looking for from other tokens?"
- The Key says, "this is what I offer to tokens looking at me."
- The Value carries, "this is what gets passed along when a match happens."
:::

::: zh
- **Query 问** ："我想从其他 token 中找到什么？"
- **Key 说** ："这是我提供给正在查看我的 token 的东西。"
- **Value 携带** ："匹配成功时，这是被传递过去的信息。"
:::

::: en
The same token plays all three roles at the same time. The Q, K, V transformations are learned matrices, so the model figures out during training what each token should look for and what it should offer.
:::

::: zh
同一个 token 同时扮演所有三个角色。Q、K、V 的转换是学习到的矩阵，所以模型在训练过程中学会每个 token 应该寻找什么、应该提供什么。
:::

::: en
Matching happens through a similarity score. Each token's Query is compared against the Key of each token it is allowed to see, using a scaled dot product. Intuitively, this measures how much the two vectors line up. The scaling keeps the numbers stable before softmax.
:::

::: zh
匹配通过 **相似度分数** 完成。每个 token 的 Query 与其允许查看的所有 token 的 Key 进行比较，使用缩放点积运算。直观上，这衡量了两个向量的对齐程度。缩放操作在 softmax 之前保持数值稳定。
:::

::: en
> **Tiny explainer: dot product**  
> A dot product is a simple way to score how aligned two vectors are. Higher alignment means a stronger match.
:::
::: zh
> **简明说明: 点积**  
> 点积是衡量两个向量对齐程度的简单方法。对齐程度越高，匹配越强。
:::

::: en
The match scores then get turned into weights using softmax. Softmax takes any set of numbers and turns them into a probability-like distribution that sums to 1. Tokens with higher match scores get higher weights, and the weights are then used to take a weighted average of the value vectors.
:::

::: zh
匹配分数通过 **softmax** 转换为权重。Softmax 将任意一组数字转换为概率分布，总和为 1。匹配分数越高的 token 获得越高的权重，然后这些权重用于对 Value 向量进行加权平均。
:::

::: en
> **Tiny explainer: softmax**  
> Softmax turns raw scores into weights that add up to 1. Big scores get big weights, small scores get small weights.
:::
::: zh
> **简明说明: Softmax**  
> Softmax 将原始分数转换为总和为 1 的权重。高分获得高权重，低分获得低权重。
:::

::: en
An example. Consider the sentence "The cat that I saw yesterday was sleeping." When the model processes "was," it needs to figure out what's doing the sleeping. The Query vector for "was" gets compared against the Key vectors of the tokens it is allowed to see. The dot product with "cat" is high, because the model has learned that verbs like "was" need a subject and that subjects like "cat" produce Key vectors that line up well. The dot product with "yesterday" is low. Softmax turns those scores into weights, "cat" gets a high weight, "yesterday" gets a low one. The model then takes a weighted sum of the corresponding value vectors, so the value for "cat" dominates the result. The new representation of "was" is now mostly shaped by the value of "cat." That's how a token several positions back becomes the referent.
:::

::: zh
举个例子。考虑句子 "The cat that I saw yesterday was sleeping"（我昨天看到的那只猫在睡觉）。当模型处理 "was" 时，它需要弄清楚是什么在睡觉。"was" 的 Query 向量与允许查看的 token 的 Key 向量进行比较。与 "cat" 的点积很高，因为模型学到像 "was" 这样的动词需要一个主语，而像 "cat" 这样的主语会产生与之很好对齐的 Key 向量。与 "yesterday" 的点积很低。Softmax 将这些分数转换为权重，"cat" 获得高权重，"yesterday" 获得低权重。然后模型对相应的 Value 向量进行加权求和，所以 "cat" 的值主导了结果。"was" 的新表示现在主要由 "cat" 的值塑造——这就是一个在几个位置之前的 token 如何成为指代对象的。
:::

::: en
There's a constraint specific to GPT-style language models, which is that they generate text left to right. A token at position 5 is only allowed to attend to positions 1 through 5. It cannot attend to tokens at positions 6, 7, 8, because those haven't been generated yet. This is called causal masking. The implementation is simple: future tokens get match scores so low they end up with effectively zero weight after softmax.
:::

::: zh
GPT 风格语言模型有一个特定约束：它们从左到右生成文本。位置 5 的 token 只能关注位置 1 到 5 的 token，不能关注位置 6、7、8 的 token，因为那些还没有生成。这叫做 **因果掩码（Causal Masking）** 。实现很简单：未来的 token 获得极低的匹配分数，在 softmax 后几乎为零权重。
:::

::: en
> **Tiny explainer: causal masking**  
> Causal masking hides future tokens. It keeps a decoder-only language model from looking ahead while predicting the next token.
:::
::: zh
> **简明说明: 因果掩码**  
> 因果掩码隐藏未来的 token。它防止仅解码语言模型在预测下一个 token 时向前看。
:::

![注意力热图：因果掩码与对"cat"的高注意力](https://0xkato.xyz/assets/transformer-attention-heatmap.png)

::: en
One of the most interesting findings in interpretability research is about specialized attention heads called induction heads, found by Anthropic in 2022. These heads learn to spot patterns of the form "A B … A" in the prompt and predict that B comes next. When the model sees "A" the second time, the induction head looks back to where "A" appeared before, sees what came after, and copies that. They're one of the clearest known mechanisms behind in-context learning, the ability of an LLM to pick up a pattern from your prompt and continue it.
:::

::: zh
可解释性研究中最有趣的发现之一是 **归纳头（Induction Head）** ，由 Anthropic 在 2022 年发现。这些专门的注意力头学会发现 prompt 中 "A B … A" 形式的模式，并预测 B 会接着出现。当模型第二次看到 "A" 时，归纳头回头查找 "A" 之前出现的位置，看后面跟着什么，然后复制那个。它们是目前已知的 **上下文学习（In-context Learning）** 背后最清晰的机制之一——即 LLM 从 prompt 中获取模式并延续它的能力。
:::

::: en
> **Tiny explainer: induction head**  
> An induction head is an attention head that notices repeated patterns in the prompt and helps continue them.
:::
::: zh
> **简明说明: 归纳头**  
> 归纳头是一种注意力头，能够注意 prompt 中的重复模式并帮助延续它们。
:::

::: en
Attention has one big cost. In full attention, each token compares against all the tokens it is allowed to see, so doubling the prompt length roughly quadruples the work. This is why long prompts are expensive to run, and why a lot of recent research is about making attention more efficient (FlashAttention, sparse attention, linear attention).
:::

::: zh
注意力有一个很大的成本。在全注意力中，每个 token 与所有允许查看的 token 进行比较，所以将 prompt 长度加倍大约会使工作量翻两番。这就是为什么长 prompt 运行成本高，以及为什么最近很多研究都致力于提高注意力效率（FlashAttention、稀疏注意力、线性注意力）。
:::

::: en
But one attention head only gives the model one learned view of those relationships.
:::

::: zh
但一个注意力头只给模型一种学习到的关系视图。
:::

---

::: en
## Multi-head Attention
:::

::: zh
## 多头注意力
:::

::: en
A single attention pass gives the model one way of deciding which tokens matter to which other tokens. That's not enough. Language has many relationships happening at the same time. Subject and verb agreement. Pronouns and the names they refer to. Long-range references between sentences. Word order and local phrases.
:::

::: zh
单次注意力传递给了模型一种决定哪些 token 对哪些其他 token 重要的方式。这还不够。语言中同时发生很多种关系：主语和动词的一致、代词和它们指代的名称、句子之间的长距离引用、词序和局部短语。
:::

::: en
Multi-head attention solves this by running attention many times in parallel, with each parallel pass operating in its own smaller space. Each parallel pass is called a head.
:::

::: zh
**多头注意力（Multi-head Attention）** 通过多次并行运行注意力来解决这个问题，每次并行传递在自己的较小空间中操作。每次并行传递叫做一个 **头（Head）** 。
:::

::: en
> **Tiny explainer: attention head**  
> An attention head is one independent attention pass with its own learned projections.
:::
::: zh
> **简明说明: 注意力头**  
> 注意力头是一次独立的注意力传递，有自己的学习投影。
:::
::: en
The part that's often described wrong, including in plenty of tutorials. Each head doesn't get a literal slice of the original token vector. Each head has its own learned projection matrices that map the full token vector down to its own smaller Q, K, and V vectors. So if a model has 4,096 numbers per token and 32 heads, each head usually works in a 128-dimensional space, but those 128 numbers are a learned projection of the full 4,096, not a fixed slice. Different "views" of the same token, not different chunks of it.
:::

::: zh
这是包括很多教程在内经常描述错误的部分。每个头并不是获得原始 token 向量的一个 literal 切片。每个头有自己的学习投影矩阵，将完整的 token 向量映射到自己的更小的 Q、K、V 向量。所以如果模型每个 token 有 4,096 个数字和 32 个头，每个头通常在 128 维空间中工作，但这 128 个数字是完整 4,096 的学习投影，而不是固定切片——是同一 token 的不同"视角"，而不是不同块。
:::

::: en
Each head runs its attention pass independently. Then the outputs of all the heads get concatenated and passed through a final linear layer that mixes them back into one full-size vector. The model learns that final mixing too.
:::

::: zh
每个头独立运行注意力传递。然后所有头的输出被拼接在一起，通过一个最终的线性层混合回一个完整大小的向量。模型也学习这个最终的混合。
:::

![多头注意力结合专门的注意力头](https://0xkato.xyz/assets/transformer-multi-head-attention.png)

::: en
What makes this interesting is that different heads often end up partially specialized. The model is never told what each head should do. Specialization emerges naturally during training. Researchers have found heads that track grammar (linking verbs to their objects, articles to their nouns), heads that figure out which pronoun refers to which name, heads that track positional patterns, induction heads, and many more. A single transformer layer might have 32 heads. A modern frontier model has dozens of layers. So a typical LLM has thousands of attention heads in total, each adding its own learned view.
:::

::: zh
有趣的是，不同的头最终往往会部分专门化。模型从未被告知每个头应该做什么。专门化是在训练中自然涌现的。研究人员发现了追踪语法（将动词与其宾语联系起来、将冠词与其名词联系起来）的头、弄清代词指代哪个名字的头、追踪位置模式的头、归纳头，以及更多。一个单独的 Transformer 层可能有 32 个头。一个现代前沿模型有几十层。所以一个典型的 LLM 总共有数千个注意力头，每个都添加了自己的学习视角。
:::

::: en
There's a practical cost concern that drove a recent architectural change. Each head needs to keep its Key and Value vectors in memory for all the tokens already generated, so that when a new token gets generated the model doesn't have to recompute everything from scratch. This is called the KV cache, and it's the main memory cost of running an LLM at long context lengths.
:::

::: zh
有一个实际成本考虑驱动了最近的架构变更。每个头需要在内存中保存所有已生成 token 的 Key 和 Value 向量，这样当生成新 token 时模型不必从头重新计算所有内容。这叫做 **KV 缓存（KV Cache）** ，这是在长上下文长度下运行 LLM 的主要内存成本。
:::

::: en
> **Tiny explainer: KV cache**  
> The KV cache stores old Key and Value vectors during generation. It saves the model from recomputing the whole prompt every time it adds a token.
:::
::: zh
> **简明说明: KV 缓存**  
> KV 缓存在生成期间存储旧的 Key 和 Value 向量。它节省模型在每次添加 token 时从头重新计算整个 prompt 的成本。
:::

::: en
Modern decoder-only LLMs mostly use a variant called Grouped-Query Attention (GQA). Instead of every head having its own keys and values, groups of heads share the same key and value heads. LLaMA-2 70B has 64 query heads but only 8 key/value heads. Mistral 7B has 32 query heads and 8 key/value heads. The result is nearly the same accuracy as full multi-head attention but with much less memory pressure and inference cost.
:::

::: zh
现代仅解码器 LLM 大多使用一种叫做 **分组查询注意力（GQA, Grouped-Query Attention）** 的变体。不是每个头都有自己的 Key 和 Value，而是多头共享相同的 Key 和 Value 头。LLaMA-2 70B 有 64 个查询头但只有 8 个 Key/Value 头。Mistral 7B 有 32 个查询头和 8 个 Key/Value 头。结果是几乎与全多头注意力相同的准确度，但内存压力和推理成本大大降低。
:::

::: en
> **Tiny explainer: GQA**  
> Grouped-Query Attention lets multiple query heads share fewer key/value heads. That cuts KV-cache memory while keeping many query views.
:::
::: zh
> **简明说明: GQA**  
> 分组查询注意力让多个查询头共享更少的 Key/Value 头。这减少了 KV 缓存内存，同时保持许多查询视角。
:::

---

::: en
## Feed-forward Network
:::

::: zh
## 前馈网络
:::

::: en
After attention finishes mixing information between tokens, every layer has a second step that nobody talks about as much. The feed-forward network.
:::

::: zh
在注意力完成 token 之间的信息混合后，每个层还有第二个步骤，这个步骤很少有人谈论—— **前馈网络（Feed-forward Network, FFN）** 。
:::

::: en
Where attention is about tokens talking to each other, the feed-forward network is about each token, on its own, doing more processing. It runs on every token's vector independently, with no cross-token mixing.
:::

::: zh
如果说注意力是 token 之间相互交流，那么前馈网络就是每个 token 独立进行更多处理。它独立地在每个 token 的向量上运行，没有跨 token 的混合。
:::

::: en
The feed-forward network does three things in order:

1. Expand the token's vector to a larger size (the original transformer used 4x, while modern SwiGLU models often use different expansion sizes).
2. Apply a non-linear function.
3. Compress the vector back down to its original size.
:::

::: zh
前馈网络按顺序做三件事：

1. 将 token 的向量扩展到更大的尺寸（原始 transformer 使用 4 倍扩展，而现代 SwiGLU 模型通常使用不同的扩展尺寸）。
2. 应用非线性函数。
3. 将向量压缩回原始尺寸。
:::

![前馈网络扩展、转换和压缩每个 token 向量](https://0xkato.xyz/assets/transformer-ffn.png)

::: en
That non-linear step in the middle is doing something specific that's worth understanding. A non-linearity is a function that bends its input. The simplest one, ReLU, outputs zero for any negative number and passes positive numbers through unchanged.
:::

::: zh
中间的非线性步骤正在做一件值得理解的具体事情。非线性是一个弯曲输入的函数。最简单的 **ReLU** ，对任何负数输出零，正数保持不变地通过。
:::

::: en
> **Tiny explainer: non-linearity**  
> A non-linearity is a function that prevents the network from collapsing into one big linear transformation.
:::
::: zh
> **简明说明: 非线性**  
> 非线性是一种防止网络崩溃为一个大线性变换的函数。
:::

::: en
Without it, the FFN would just be two linear layers stacked together, and stacking pure linear math collapses. Two linear layers in a row are mathematically equivalent to a single linear layer, and a hundred linear layers in a row are still equivalent to one. The non-linearity is what stops that collapse, and it's the reason the FFN can do something richer than a single matrix multiplication.
:::

::: zh
没有它，FFN 只是两个线性层堆叠在一起，而纯线性数学的堆叠会崩溃。两层线性相继在数学上等价于一层线性，一百层线性相继也还是等价于一层。非线性阻止了这种崩溃，这就是 FFN 能够做比单个矩阵乘法更丰富的事情的原因。
:::

::: en
The original transformer used ReLU. GPT and BERT moved to GELU. Modern models like LLaMA, Mistral, and PaLM use SwiGLU. The expand-then-compress structure stayed the same. The non-linearity itself is what's been iterated on.
:::

::: zh
原始 transformer 使用 ReLU。GPT 和 BERT 转向 GELU。现代模型如 LLaMA、Mistral 和 PaLM 使用 SwiGLU。扩展-压缩结构保持不变，迭代的是非线性函数本身。
:::

::: en
Most of the parameters in a dense transformer model live in the FFN, not in attention. A large share of the weights sit in feed-forward layers.
:::

::: zh
密集 transformer 模型的大部分参数在 FFN 中，而不是在注意力中。大部分权重位于前馈层中。
:::

::: en
And those parameters aren't generic. They're where much of the model's stored factual and semantic structure lives. Researchers have found that some neurons inside the FFN are strongly associated with specific concepts or facts. One neuron might activate strongly on Eiffel-Tower-related text. Another on programming languages. Another on past-tense verbs. When a model "knows" that Paris is the capital of France, that fact is represented across FFN weights and activations in specific layers.
:::

::: zh
而且这些参数不是通用的。它们是模型存储的事实和语义结构的大部分所在。研究人员发现 FFN 内部的一些神经元与特定概念或事实强烈相关。一个神经元可能在与埃菲尔铁塔相关的文本上强烈激活，另一个可能在编程语言上，另一个可能在过去时态动词上。当模型"知道"巴黎是法国的首都时，这个事实在特定层的 FFN 权重和激活中表示。
:::

::: en
This stored-memory property has an interesting consequence. Researchers have figured out how to directly edit some facts in a trained model without retraining it. Methods like ROME (Rank-One Model Editing) can change "the Eiffel Tower is in Paris" to "the Eiffel Tower is in Rome" by making a targeted low-rank edit to a specific FFN weight matrix. The model then tends to generate text consistent with the edited association.
:::

::: zh
这种存储记忆特性有一个有趣的副产品。研究人员已经弄清楚如何在不重新训练的情况下直接编辑训练模型中的一些事实。 **ROME（Rank-One Model Editing）** 等方法可以通过对特定 FFN 权重矩阵进行针对性的低秩编辑，将"埃菲尔铁塔在巴黎"改为"埃菲尔铁塔在罗马"。然后模型倾向于生成与编辑后的关联一致的内容。
:::

::: en
Some modern frontier models have started replacing the dense FFN with something called Mixture of Experts (MoE). Instead of one feed-forward network per layer, the model has many parallel FFNs (called experts) and a tiny router network that picks which experts process each token. Mixtral 8x7B has 8 experts per layer; only 2 are activated for any given token. The total parameter count goes up substantially, but the compute per token grows much more slowly because only a few experts run. That's how you scale parameter count without scaling inference cost in proportion.
:::

::: zh
一些现代前沿模型开始用叫做 **专家混合（MoE, Mixture of Experts）** 的东西替换密集 FFN。不是每层一个前馈网络，而是有许多并行的 FFN（叫做专家）和一个选择哪些专家处理每个 token 的小型路由网络。Mixtral 8x7B 每层有 8 个专家；对任何给定 token 只有 2 个被激活。总参数计数大幅增加，但每个 token 的计算增长慢得多，因为只有少数专家运行。这就是参数计数增长而不使推理成本按比例增长的方式。
:::

::: en
> **Tiny explainer: MoE**  
> Mixture of Experts means the model has several feed-forward networks and routes each token through only a few of them.
:::
::: zh
> **简明说明: MoE**  
> 专家混合意味着模型有多个前馈网络，并将每个 token 路由到仅其中几个运行。
:::

::: en
Mixtral 8x7B has 46.7 billion total parameters but uses about 12.9 billion per token. This has become a common option for very large models because it lets you keep growing the parameter count without making inference cost grow in proportion.
:::

::: zh
Mixtral 8x7B 总参数为 467 亿，但每个 token 约使用 129 亿。这已成为超大模型的常见选择，因为它允许继续增加参数计数而不使推理成本按比例增长。
:::

---

::: en
## Residual Stream and Layer Normalization
:::

::: zh
## 残差流与层归一化
:::

::: en
The residual stream is what makes the model "additive" instead of "replacing." After attention runs, or after the feed-forward network runs, the result usually doesn't replace the token's vector. It gets added to it. Position by position. The new vector equals the old vector plus the sub-block's output.
:::

::: zh
**残差流（Residual Stream）** 使模型是"加性的"而不是"替换性的"。注意力运行后，或者前馈网络运行后，结果通常不替换 token 的向量，而是被加到它上面。逐位置加。新向量等于旧向量加上子块的输出。
:::

::: en
> **Tiny explainer: residual connection**  
> A residual connection adds a block's output back to the vector it started from. It gives information and gradients a shortcut through the network.
:::
::: zh
> **简明说明: 残差连接**  
> 残差连接将块的输出加回到它开始的向量上。它为信息和梯度提供通过网络的快捷路径。
:::

::: en
Across thirty or fifty or a hundred layers, each layer's contribution accumulates instead of simply overwriting the previous vector. That running sum is called the residual stream, and it has a strange property. The original input embeddings still have a direct additive path into late layers, mixed together with every sub-block's contribution along the way.
:::

::: zh
经过三十层、五十层或一百层，每个层的贡献累积而不是简单地覆盖前面的向量。这个运行的和叫做残差流，它有一个奇怪的特性。原始输入嵌入仍然有直接加性路径到后面的层，与沿途每个子块的贡献混合在一起。
:::

![残差流累积注意力和前馈输出](https://0xkato.xyz/assets/transformer-residual-stream.png)

::: en
Residual connections weren't invented for transformers. They came from ResNet (He et al. 2015), originally for image recognition. The motivation was that deep networks were impossible to train. The training signal got too weak (or sometimes too strong) by the time it traveled back through many layers. The model couldn't actually learn from its own mistakes. Adding a shortcut path let the signal flow directly back from the output to the input. Suddenly you could train networks with hundreds of layers. Transformers inherited the same trick.
:::

::: zh
残差连接不是为 transformer 发明的。它们来自 ResNet（He et al., 2015），最初用于图像识别。动机是深度网络难以训练。训练信号在回传多层后变得太弱（有时太强）。模型实际上无法从自己的错误中学习。添加快捷路径让信号直接从输出流回输入。突然之间你可以训练有数百层的网络了。Transformer 继承了同样的技巧。
:::

::: en
In modern interpretability research, the residual stream has become the central object. Every component, every attention head, every feed-forward network, even the unembedding step at the end, reads from the residual stream and writes back to it.
:::

::: zh
在现代可解释性研究中，残差流已成为核心对象。每个组件、每个注意力头、每个前馈网络，甚至最后的去嵌入步骤，都从残差流读取并写回。
:::

::: en
The second piece, layer normalization, exists for a much more practical reason. Without it, the residual stream would not stay stable. Numbers flowing through dozens of additions tend to either explode upward or collapse toward zero. Either way, training fails. Layer normalization rescales each token's vector back into a controlled range between sub-blocks.
:::

::: zh
第二个组件—— **层归一化（Layer Normalization）** ——存在的原因更加实际。没有它，残差流就不会保持稳定。通过数十次加法的数字往往要么向上爆炸，要么向零崩溃。两种情况都会导致训练失败。层归一化在子块之间将每个 token 的向量重新缩放到受控范围内。
:::

::: en
> **Tiny explainer: layer normalization**  
> Layer normalization rescales a token vector so its numbers stay in a stable range while the model trains.
:::
::: zh
> **简明说明: 层归一化**  
> 层归一化重新缩放 token 向量，使其数值在模型训练时保持在稳定范围内。
:::

::: en
The original 2017 transformer applied normalization AFTER each sub-block (post-norm). This worked for shallow models but became harder to train reliably as depth increased. Modern transformers (GPT-2 onward, LLaMA, Mistral) commonly apply normalization BEFORE each sub-block (pre-norm). That's one of the changes that made very deep transformers easier to train.
:::

::: zh
原始 2017 transformer 在每个子块之后应用归一化（post-norm 后归一化）。这对浅层模型有效，但随着深度增加变得越来越难可靠地训练。现代 transformer（GPT-2 之后、LLaMA、Mistral）通常在每个子块之前应用归一化（pre-norm 前归一化）。这是使非常深的 transformer 更容易训练的变更之一。
:::

::: en
The function itself has also changed. Many modern open models (LLaMA, Mistral, Gemma, Phi) use a simpler variant called RMSNorm. The original layer normalization did two things at once: shift each vector toward zero, then rescale the size of the numbers. RMSNorm drops the shift step and keeps only the rescaling. Empirically, the rescaling carries most of the benefit while being cheaper to compute.
:::

::: zh
函数本身也改变了。许多现代开源模型（LLaMA、Mistral、Gemma、Phi）使用一个更简单的变体叫做 **RMSNorm** 。原始层归一化同时做两件事：将每个向量移向零，然后重新缩放数字大小。RMSNorm 放弃了移位步骤，只保留重新缩放。根据经验，重新缩放带来大部分好处，同时计算成本更低。
:::

::: en
> **Tiny explainer: RMSNorm**  
> RMSNorm is a cheaper normalization method that rescales vector size without subtracting the mean first.
:::
::: zh
> **简明说明: RMSNorm**  
> RMSNorm 是一种更便宜的归一化方法，只重新缩放向量大小，而不先减去均值。
:::

::: en
So that's the unglamorous machinery. Without residual connections, very deep models become much harder to train. Without layer normalization, the running sum can blow up or collapse. With both, you get models hundreds of layers deep.
:::

::: zh
这就是不 glamorous 的 machinery。没有残差连接，非常深的模型变得难训练得多。没有层归一化，运行和可能爆炸或崩溃。有了两者，你可以得到数百层深的模型。
:::

---

::: en
## Next-token Prediction
:::

::: zh
## 下一个 Token 预测
:::

::: en
After all the layers of attention and feed-forward processing finish, the model has a vector for each token in the sequence. During generation, to predict the next word, it takes the final vector of the last token only.
:::

::: zh
在所有层的注意力和前馈处理完成后，模型对序列中的每个 token 都有一个向量。在生成期间，为了预测下一个词，它只使用最后一个 token 的最终向量。
:::

::: en
That last vector gets converted into one number per possible next token. If the vocabulary has 100,000 tokens, that's 100,000 numbers. These numbers are called logits. They aren't probabilities yet. They can be any size, positive or negative.
:::

::: zh
最后一个向量被转换为每个可能下一个 token 对应一个数字。如果词表有 100,000 个 token，那就是 100,000 个数字。这些数字叫做 **logits** 。它们还不是概率，可以是任意大小的正数或负数。
:::

::: en
> **Tiny explainer: logits**  
> Logits are raw scores for each possible next token. They become probabilities only after softmax.
:::
::: zh
> **简明说明: Logits**  
> Logits 是每个可能下一个 token 的原始分数。它们在 softmax 之后才变成概率。
:::

::: en
A softmax turns those logits into the model's probability distribution over possible next tokens. Same operation as before, different place in the model.
:::

::: zh
Softmax 将这些 logits 转换为模型在可能下一个 token 上的概率分布。与之前相同的操作，但在模型的不同位置。
:::

::: en
The model usually does not just pick the highest-probability token every time. Decoding settings control how deterministic or varied the output is. Temperature changes how sharp the distribution is. Top-k and top-p limit the choices to the most plausible next tokens. That is why the same model can feel precise in one setting and more creative in another.
:::

::: zh
模型通常不是每次都只选择最高概率的 token。解码设置控制输出的确定性或多样性程度。 **Temperature（温度）** 改变分布的锐度。 **Top-k** 和 **Top-p** 将选择限制在最可能的下一个 token 上。这就是为什么同一模型在一个设置下感觉精确，在另一个设置下更有创造力。
:::

::: en
> **Tiny explainer: temperature**  
> Temperature controls randomness during sampling. Low temperature makes the model more conservative; high temperature makes it more varied.
:::

::: zh
> **Tiny explainer: temperature**  
> 温度控制采样时的随机性。低温使模型更保守，高温使模型更多样。
:::

::: en
Once a token is picked, it gets added to the input. The model runs the next step on the longer sequence, usually reusing the KV cache so it doesn't recompute the whole prefix from scratch. New attention for the new token. New feed-forward. New final vector. New prediction. The loop continues until the model emits an end-of-sequence token or hits a length limit. A whole paragraph is just this loop, one token at a time.
:::

::: zh
一旦选择了一个 token，它就被添加到输入中。模型在更长的序列上运行下一步，通常重用 KV 缓存，这样不必从头重新计算整个前缀。新 token 的新注意力。新前馈。新的最终向量。新的预测。循环继续直到模型发出序列结束 token 或达到长度限制。一整段文字就是这样一次一个 token 地循环这个过程。
:::

::: en
This single objective, predicting the next token, is the core training signal for a base LLM. The base model isn't trained on factual accuracy, conversational ability, reasoning, or coding directly. It's trained to predict the next token in massive amounts of text. Later post-training can then tune the model for instruction following, preference, safety, and conversational behavior.
:::

::: zh
这个单一目标——预测下一个 token——是基础 LLM 的核心训练信号。基础模型不是直接训练事实准确性、对话能力、推理或编码的。它是在海量文本上训练预测下一个 token。后续的后训练可以微调模型以遵循指令、偏好、安全性和对话行为。
:::

::: en
There's been a major efficiency innovation worth knowing about. It's called speculative decoding. A small fast model proposes several tokens ahead. The big model verifies them in parallel. If the proposed tokens are accepted under the big model's probabilities, accept them. If not, fall back to the big model. Done correctly, the output distribution matches running the big model alone, but the loop can run much faster.
:::

::: zh
有一个值得知道的主要效率创新。它叫做 **推测解码（Speculative Decoding）** 。一个小型快速模型一次提出几个 token。大型模型并行验证它们。如果提议的 token 在大模型的概率下被接受，就接受。如果不接受，就回退到大模型。如果做得正确，输出分布与单独运行大模型相同，但循环可以运行得快得多。
:::

::: en
> **Tiny explainer: speculative decoding**  
> Speculative decoding uses a small draft model to guess ahead, then asks the larger model to verify several guessed tokens at once.
:::

::: zh
> **Tiny explainer: speculative decoding**  
> 推测解码使用小型草稿模型提前猜测，然后让大模型一次验证多个猜测的 token。
:::

::: en
The next-token prediction loop is the simplest part of the architecture, but it's what makes the whole thing work.
:::

::: zh
下一个 token 预测循环是架构中最简单的部分，但正是它使整个系统工作起来。
:::

---

::: en
## Architecture vs Trained Weights
:::

::: zh
## 架构 vs 训练权重
:::

::: en
We've gone through the core mechanisms: tokens, embeddings, positional encoding, attention, multi-head attention, the feed-forward network, the residual stream and normalization, and the next-token loop on the output side. That's the basic architecture in one pass.
:::

::: zh
我们已经过了一遍核心机制：token、嵌入、位置编码、注意力、多头注意力、前馈网络、残差流和归一化，以及输出端的下一个 token 循环。这是一次传递中的基本架构。
:::

::: en
So what's actually different between GPT and Claude and Gemini and LLaMA? Public details vary, and the proprietary models do not publish every architectural choice. But at the level this post is covering, they broadly sit in the same transformer-family design space.
:::

::: zh
那么 GPT、Claude、Gemini 和 LLaMA 之间实际有什么不同？公开细节各不相同，专有模型也不会公布每个架构选择。但在这个帖子所涵盖的水平上，它们大致处于相同的 transformer 家族设计空间。
:::

::: zh
大多数现代基于 transformer 的 LLM 使用相同的宽泛结构：分词、嵌入、位置编码、堆叠的 Transformer 层（每层有多头注意力和一个前馈网络）、残差流、层归一化和下一个 token 预测。
:::

::: en
What changes between models is:

1. The trained weights themselves, learned from different training data at different scales.
2. The configuration: number of layers, vocabulary size, head count, parameter count, MoE or dense.
3. The post-training: instruction tuning, learning from human feedback, safety controls applied on top of the base model.
:::

::: zh
模型之间变化的是：

1. **训练权重本身** ：从不同规模的不同训练数据中学习到的。
2. **配置** ：层数、词表大小、头数、参数数量、MoE 或密集。
3. **后训练** ：指令微调、从人类反馈中学习、在基础模型之上应用的安全控制。
:::

::: en
> **Tiny explainer: weights**  
> Weights are the learned numbers inside the model. Training changes those numbers until the model predicts text well.
:::

::: zh
> **Tiny explainer: weights**  
> 权重是模型内部学习到的数字。训练会改变这些数字，直到模型能够很好地预测文本。
:::

---

::: zh
> **原文链接** ：[How LLMs Actually Work](https://www.0xkato.xyz/how-llms-actually-work/)
> **译者** ：AI 辅助翻译 + 人工校对
> **本文采用** ：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
:::
