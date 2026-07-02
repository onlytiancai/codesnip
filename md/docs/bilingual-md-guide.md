# 双语 Markdown 翻译规范

> 用途：把一份英文 markdown 文档翻译成"中英对照"格式，供后续 AI 翻译 / 校对 / 阅读使用。
> 对应脚本：`scripts/check_bilingual_md.py`

---

## 1. 文件命名

| 文件 | 命名 |
| --- | --- |
| 原文 | `name.md` |
| 双语版 | `name-zh-CN.md` |

例：`how-llms-actually-work.md` → `how-llms-actually-work-zh-CN.md`

---

## 2. 核心结构：块标记

整篇文档由 `::: en` / `::: zh` 两种块交替组成。每个块用 `:::` 关闭。

最小骨架：

```markdown
::: en
English content
:::

::: zh
中文内容
:::
```

**重要规则**

1. `::: en` / `::: zh` / `:::` 都必须**独占一行**（行首到行尾只有标记本身）。
2. 英文块和中文块**一一对应**，数量必须相等，顺序应保持 en → zh → en → zh …。
3. 块外允许出现的内容**只有图片和代码块**（见 §4）。

---

## 3. 各元素翻译规则

### 3.1 标题（`#` ~ `######`）

英文和中文各包一个**同级**标题，层级和数量必须一致。

```markdown
::: en
## Tokenization
:::

::: zh
## 分词
:::
```

### 3.2 段落

每个英文段落对应一段中文。空行分隔段落，所以块内空行的位置也必须对齐。

```markdown
::: en
First paragraph.

Second paragraph.
:::

::: zh
第一段。

第二段。
:::
```

### 3.3 列表

列表项逐条对应；`1.`/`-` 等标记样式保持一致。

```markdown
::: en
1. The trained weights themselves.
2. The configuration.
3. The post-training.
:::

::: zh
1. 训练权重本身。
2. 配置。
3. 后训练。
:::
```

### 3.4 引用（`> ...`）

整块引用整体翻译；Tiny explainer / 译者注等特殊引用框按同样规则。

```markdown
::: en
> **Tiny explainer: token ID**
> A token ID is the integer the model uses.
:::

::: zh
> **简明说明：词元 ID**
> 词元 ID 是模型使用的整数。
:::
```

### 3.5 表格

逐格翻译。分隔行（`| --- |`）保持原样。

```markdown
::: en
| Model | Layers |
| --- | --- |
| A    | 12     |
:::

::: zh
| 模型 | 层数 |
| --- | --- |
| A   | 12   |
:::
```

### 3.6 粗体 / 斜体 / 行内代码 / 链接

- 粗体、斜体、行内代码标记 (`**` / `*` / `` ` ``) 保留。
- 链接 `[text](url)` 的 URL 保留，`text` 翻译。
- 专有名词、英文术语**首次出现时附原文**（如"分词（Tokenization）"）。

### 3.7 图片（无需翻译）

图片**不需要**包在 `::: en` / `::: zh` 里，可直接放在块外。两侧都用同一张图，避免重复。

```markdown
::: en
Some English paragraph.
:::

![](https://example.com/image.png)

::: zh
对应的中文段落。
:::
```


### 3.8 代码块（无需翻译）

代码块也放在**块外**，原文与译文两侧共享同一段代码，不重复。

````markdown
::: en
Run this command:

```bash
pnpm install
```
:::

````

> 说明：上面的双层 ```` ``` ```` 只是为了在文档里展示代码块语法。实际文件中用单层 ```。

### 3.9 水平分割线 `---`

文档级分割线（如分章处的 `---`）放**块外**，不要包在 `::: en/zh` 里。

---

## 4. 不放进块内的内容汇总

| 内容 | 是否进块 | 说明 |
| --- | --- | --- |
| 图片 `![alt](url)` | ❌ | 两侧共用，块外即可 |
| 代码块 ` ``` ` | ❌ | 两侧共用，块外即可 |
| 水平分割线 `---` | ❌ | 文档级分隔 |
| 文件级元信息 | ⚠️ | 通常放最前面，按 `::: en` / `::: zh` 各包一份 |

---

## 5. 完整示例

下面是一段**完整正确**的双语段落（标题 + 段落 + 引用 + 列表 + 块外图片）：

```markdown
::: en
## Tokenization

Models don't read text directly. They read integer IDs.

> **Tiny explainer: token ID**
> A token ID is the integer the model uses.

The most common pieces become single tokens.
:::

::: zh
## 分词

模型不直接读取文本，而是读取整数 ID。

> **简明说明：词元 ID**
> 词元 ID 是模型使用的整数。

最常见的片段会作为单个 token。
:::

![分词示意图：文本被切分为 token ID 序列](https://example.com/tokenization.png)
```

---

## 6. 常见错误

| 错误 | 后果 | 怎么修 |
| --- | --- | --- |
| 缺一个 `:::` 关闭 | 后续所有内容都被吞进同一块 | 数清楚 `::: en/zh` 和 `:::` 数量 |
| 中英块数量不等 | 出现"孤儿"段落 | 重新对照原文，逐节补齐 |

---

## 7. 验证

翻译完成后，跑：

```bash
python scripts/check_bilingual_md.py path/to/file-zh-CN.md
```

脚本会输出：

- en/zh 块数量及是否平衡
- 中英文两侧的：H1-H6 标题数、段落数、列表项数、表格数、引用数、图片数、代码块数
- 一致性检查：发现数量不一致的元素会标 ⚠️

**判定标准**

- ✅ en/zh 块数量相等
- ✅ 同级标题数量相等
- ✅ 段落数差 ≤ 1（少数情况如元信息块可容差）
- ⚠️ 表格、引用、列表项数不一致 → 回去检查
- 图片 / 代码块数两侧应一致（共享同一资源）
