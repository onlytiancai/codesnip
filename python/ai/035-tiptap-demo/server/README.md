# tiptap-rewrite-server

中英翻译 + 英语写作 AI 助手的后端。包装本地 [Ollama](https://ollama.com/) 模型，提供翻译（带语气）、单词/短语替换、整句改写、语法检查、自定义指令改写五个能力。

## 启动

```bash
cp .env.example .env       # 按需改 OLLAMA_MODEL
pnpm install
pnpm dev                   # tsx watch，监听 8787
```

可用模型示例：`s2021008840/hy-mt2:1.8b`（默认）。前提：本地 Ollama 已启动并已 `ollama pull` 该模型。

## 手工测试 CLI

不用起 server、不用开浏览器，直接调 service 层：

```bash
cd server
pnpm cli --help
pnpm cli selection --text "very important" --kind phrase --context "Education is very important for children."
pnpm cli sentence  --text "Education is very important for children." --operation rewrite --tone formal
pnpm cli grammar   --text "He go to school last yesterday."
pnpm cli instruction --text "The results were kind of surprising." --instruction "Make it sound more natural."
echo "教育对孩子很重要。" | pnpm cli translate
pnpm cli models
```

命令与 `src/services/ollama.ts` 的方法一一对应：`selection` → `rewriteSelection`、`sentence` → `rewriteSentence`、`grammar` → `grammarCheck`、`instruction` → `rewriteInstruction`，另外有 `translate`（TranslateService）、`raw`（直接发 system/user，不做任何解析）、`models`（列出本地模型）。

调试用的三个开关：

| 开关 | 作用 |
|---|---|
| `--prompt` | 打印**实际发出**的 system / user 消息（不是重新拼的，是录到的真实请求） |
| `--raw` | 打印模型原始输出，绕过 `parseCandidates` / `cleanText`——解析失败时用它看是哪一步炸的 |
| `--json` | 把 command / model / prompt / raw / result / error / 耗时一次性输出成 JSON |

其他选项：`--model`（默认 `OLLAMA_MODEL`，再默认 `s2021008840/hy-mt2:1.8b`）、`--host`（默认 `http://127.0.0.1:11434`）、`--text` / `--file` / 位置参数 / stdin 管道 / 交互输入（按此优先级取输入）。

CLI 通过给 `createOllamaService` 注入一个「录音 client」实现——它包住真实 `client.chat`，把请求和原始响应记下来再转发，所以测的就是生产代码本身的那条路径。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `OLLAMA_MODEL` | `s2021008840/hy-mt2:1.8b` | Ollama 模型名 |
| `PORT` | `8787` | Web 端口 |
| `MAX_CANDIDATES` | `5` | 候选上限（实际取模型返回值与 max 较小者） |

> `OLLAMA_HOST` 不读取——直接走 `ollama` npm 包的默认值（`http://127.0.0.1:11434`）。要改请在 `src/services/ollama.ts` 里注入 client。

## 语气

`neutral`（中性）/ `formal`（正式）/ `casual`（口语）/ `professional`（专业）/ `friendly`（友好）/ `concise`（简洁）/ `detailed`（详细）/ `persuasive`（有说服力）。

`tone` 是**可选**字段：不传时走各自的默认 prompt，输出与加语气之前逐字节一致；传了才在 system prompt 末尾追加一行 `Write in a <tone> tone: <guidance>.`。枚举定义在 `src/prompts/tones.ts`（`TONE_IDS` / `TONE_GUIDANCE` / `isToneId`），`/api/translate` 与 `/api/rewrite-sentence` 共用。传了枚举外的值 → `400`。

## API

### `POST /api/translate`

请求体：
```json
{ "text": "教育对孩子很重要。", "source": "zh", "target": "en", "tone": "formal" }
```

`source` / `target` / `tone` 都可省略（目前只支持 `zh → en`）。

成功响应 `200`：
```json
{ "translation": "Education is of great importance to children." }
```

错误码：

| 状态码 | 含义 |
|---|---|
| `400` | 缺 `text` / `text` 非字符串或为空 / `source`、`target` 不是 `zh`/`en` / `tone` 不在枚举内 |
| `500` | 模型返回了无法解析的文本 |
| `502` | Ollama 上游调用失败（ECONNREFUSED 等） |

### `POST /api/rewrite-selection`

单词或短语 → 候选数组。`kind` 决定 prompt 的约束强度：

- `word`：候选必须是**同一词性**，且不能只是词形变化（`big` 不给 `bigger`）
- `phrase`：候选**可以是多词短语**，但必须填进原文相同的语法槽位；prompt 明确要求**搭配正确、拼接后语法正确、与前后文连贯**，并优先更紧凑的替代而不是长句重写

请求体：
```json
{ "text": "very important", "kind": "phrase", "context": "Education is very important for children.", "sourceSentence": "教育对孩子很重要。" }
```

`context` / `sourceSentence` 可省略。

成功响应 `200`：
```json
{ "candidates": ["essential", "of great importance", "crucial"] }
```

错误码：

| 状态码 | 含义 |
|---|---|
| `400` | 缺 `text` / `text` 为空 / 缺 `kind` 或 `kind` 不是 `word`\|`phrase` / `context`、`sourceSentence` 非字符串 |
| `500` | 模型返回了无法解析为 JSON 数组的内容 |
| `502` | Ollama 上游调用失败 |

### `POST /api/rewrite-sentence`

整句改写，四个 operation：

| operation | 含义 |
|---|---|
| `rewrite` | 换措辞和句式，表达同一意思；**只有它接受 `tone`** |
| `improve` | 修语法、生硬搭配、用词和地道表达；意思和长度基本不变 |
| `shorten` | 明显变短，砍掉填充词和限定词，但保留核心论点 |
| `expand` | 补上读者需要的信息：支撑细节、具体例子、明说的后果；不许注水 |

请求体：
```json
{ "sentence": "Education is very important for children.", "operation": "rewrite", "tone": "formal", "context": "...", "sourceSentence": "..." }
```

成功响应 `200`：
```json
{ "text": "Education is of paramount importance to children." }
```

错误码：

| 状态码 | 含义 |
|---|---|
| `400` | 缺 `sentence` / 为空 / `operation` 不在四个枚举内 / `tone` 不在枚举内 / `context`、`sourceSentence` 非字符串 |
| `500` | 模型返回了无法解析的文本 |
| `502` | Ollama 上游调用失败 |

### `POST /api/grammar-check`

**只检查，不改写。** 后端把模型的回答**原样透传**，前端也**原样显示**——不做任何结构化解析。

请求体：
```json
{ "sentence": "He go to school last yesterday." }
```

成功响应 `200`：
```json
{
  "text": "「go」应该改成「goes」，因为主语 He 是第三人称单数。另外「last」和「yesterday」意思重复，去掉一个就行。"
}
```

`text` 就是模型写的原文（前端渲染成 `<pre class="ai-menu__grammar">`）。没有错误时模型回「这句话没有语法或标点错误。」

**为什么连格式也不要：** 本地默认模型 `s2021008840/hy-mt2:1.8b` 只有 1.8B，让它吐结构化 JSON 不可靠——一旦解析失败，整句话会塌成一句笼统的 `cannot parse JSON object from model response`，把「模型回了段自然语言」这件事完全掩盖掉。改回固定管道格式、去掉 JSON、去掉 `suggestion` 之后仍然会偶发失败。所以最终：user message 就是**原句本身**（不加任何 `Sentence:` 之类的前缀），system prompt 让模型像人一样用中文写一两句大白话，读起来最自然，前端原样显示。

系统 prompt 显式禁止报风格偏好（`Do not comment on style, tone, vocabulary, or word choice`）——这一句是实测出来的，不写的话小模型会大量输出「我觉得这里可以更地道」这类废话。

错误码：

| 状态码 | 含义 |
|---|---|
| `400` | 缺 `sentence` / `sentence` 非字符串或为空 |
| `502` | Ollama 上游调用失败 |

> **换模型建议**：实测 `hy-mt2`（翻译模型）做语法检查基本不可用——它会把 system prompt 里的示例抄回来当答案，还会对本来正确的句子报「缺少句末句号」。`gemma4:e2b` 最干净，`gemma3:4b-it-qat` 可接受。改 `OLLAMA_MODEL` 即可。

### `POST /api/rewrite-instruction`

按用户自由指令改写任意文本（词、短语、整句都行）。

请求体：
```json
{ "text": "The results were kind of surprising.", "instruction": "Make it sound more natural.", "context": "..." }
```

成功响应 `200`：
```json
{ "text": "The results came as quite a surprise." }
```

错误码：

| 状态码 | 含义 |
|---|---|
| `400` | 缺 `text` / 缺 `instruction` / 任一为空 / `context` 非字符串 |
| `500` | 模型返回了无法解析的文本 |
| `502` | Ollama 上游调用失败 |

### `GET /healthz`

冒烟用，返回 `{ "ok": true }`。

## 设计约定

**三层结构**：`prompts/`（纯函数，无 IO，可单测）→ `services/`（工厂，client 可注入）→ `routes/`（校验 + 状态码分类）。

每个能力在 `prompts/` 里都拆成 `buildXxxSystem()` + `buildXxxPrompt()`（+ 需要时才有的 `parseXxx()`）。**system 和 user 分开**很重要：语气、操作类型、选区类型这类硬约束走 system message（模型对 system 的服从度更高），不要塞进 user message 再配一个泛泛的 system——早期版本就是这么写的，prompt 重复且互相打架。

`prompts/cleanText.ts` 是翻译 / 整句改写 / 自定义指令三个纯文本能力的共享清洗器：剥 markdown 围栏 → 剥 `翻译：`/`Rewritten:` 这类前缀 → 剥首尾成对引号 → 拒绝空串。**语法检查不过这个清洗器**——它的输出就是给用户看的原文，清洗只会改掉模型想说的东西。

`parseCandidates` 用候选降级：整段 JSON → ``` 围栏里的 JSON → 正文里的 `[...]`。关键是**保留第一个"能 parse 但形状不对"的错误**（`shapeError`），只在所有候选都失败时才抛它——否则「模型返回了对象但里面没有数组」会被一句笼统的 "cannot parse JSON" 盖掉，排查起来很痛苦。语法检查曾经也走这套逻辑，现在整个删掉了（见上面 `/api/grammar-check` 一节）。

`routes/rewrite.ts` 里 `classify(err)` 靠错误消息里的关键词区分解析失败（500）和上游失败（502）。

DI：`createApp({ translateService, aiService })`。测试注入 mock service，不需要启动真 Ollama。

## 测试

```bash
pnpm test            # 一次性
pnpm test:watch      # watch
pnpm type-check      # tsc --noEmit
```

测试用 `supertest` + `vitest`，通过 `tests/helpers.ts` 的 `buildApp()` 注入 mock service 打到真实路由层，无需启动真 Ollama。

| 文件 | 用例 | 覆盖 |
|---|---|---|
| `tests/matchCase.test.ts` | 5 | 大小写保留 |
| `tests/tones.test.ts` | 8 | 语气枚举 + guidance 进 prompt，无 tone 时逐字节不变 |
| `tests/translate.test.ts` | 13 | `/api/translate` 路由，含 tone 白名单 400 |
| `tests/translateWord.test.ts` | 10 | `cleanText` / `parseTranslation` 各分支 |
| `tests/rewriteSelection.test.ts` | 16 | word vs phrase 的 prompt 差异 + `parseCandidates` 降级 |
| `tests/rewriteSentence.test.ts` | 14 | 四个 operation 的 prompt + tone 只作用于 rewrite |
| `tests/grammarCheck.test.ts` | 7 | prompt 断言（原句直传、无输出格式、禁报风格偏好） |
| `tests/rewriteInstruction.test.ts` | 8 | 指令拼进 prompt + 上下文不进输出 |
| `tests/rewriteRoutes.test.ts` | 29 | 四个端点的 400 / 500 / 502 路径 |
| `tests/services.test.ts` | 9 | 断言 prompt 真的进了 `client.chat` 的 messages，语法检查原样返回 |

合计 **119 个用例 / 10 个文件**。

## 目录

```
server/
├── src/
│   ├── index.ts                        # 入口，listen；createOllamaService({ model })
│   ├── app.ts                          # createApp 工厂，挂 translate + rewrite router
│   ├── routes/
│   │   ├── translate.ts                # POST /api/translate
│   │   └── rewrite.ts                  # 4 个端点：selection / sentence / grammar / instruction
│   ├── services/
│   │   ├── translate.ts                # TranslateService（client 可注入）
│   │   └── ollama.ts                   # AiService：4 个方法 + 私有 chat()，client 可注入
│   ├── cli.ts                          # 手工测试 CLI（直接调 service，可打印 prompt / raw）
│   ├── prompts/
│   │   ├── tones.ts                    # TONE_IDS / TONE_GUIDANCE / toneInstruction
│   │   ├── cleanText.ts                # 共享清洗：围栏 / 前缀 / 引号
│   │   ├── translate.ts                # system + prompt + parse
│   │   ├── rewriteSelection.ts         # word / phrase 候选
│   │   ├── rewriteSentence.ts          # rewrite / improve / shorten / expand
│   │   ├── grammarCheck.ts             # 纯文本格式的语法检查（不解析，原样透传）
│   │   └── rewriteInstruction.ts       # 自由指令改写
│   └── utils/matchCase.ts              # Good→Great / good→great / GOOD→GREAT
└── tests/                              # vitest，10 个文件 119 个用例
```

server/curl-tests.sh     (executable)

用法：
```
bash curl-tests.sh              # 全套冒烟（需本地 Ollama 已启动）
bash curl-tests.sh healthz      # 仅 /healthz
bash curl-tests.sh translate    # /api/translate：无 tone / 有 tone
bash curl-tests.sh selection    # /api/rewrite-selection：word / phrase
bash curl-tests.sh sentence     # /api/rewrite-sentence：4 个 operation + tone
bash curl-tests.sh grammar      # /api/grammar-check（返回 {text}，模型原文）
bash curl-tests.sh instruction   # /api/rewrite-instruction
bash curl-tests.sh errors       # 4 个端点的错误路径（11 个 case）
BASE=http://host:port bash curl-tests.sh   # 改 base URL
```
