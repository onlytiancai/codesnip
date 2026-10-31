# tiptap-rewrite-server

中英双向翻译 + 英语写作 AI 助手 + 本地词库查词的后端。提供翻译（中译英 / 英译中，带语气）、单词/短语替换、整句改写、语法检查、自定义指令改写五个 AI 能力，外加一个纯本地的查词服务。

**LLM 后端可换**：本地 Ollama、任何 OpenAI 兼容端点（OpenAI / DeepSeek / Moonshot / Qwen / vLLM…）、Anthropic Claude。每个 provider 各自带一套提示词 —— 因为不同模型需要的措辞不一样（小模型要直白的硬规则，大模型要松弛的指令 + 结构化输出）。

**提示词全部落在 JSON 里**，可以在 Web UI 上改，改动留历史、能回滚。

## 启动

```bash
pnpm install
pnpm dev                   # tsx watch，监听 8787
```

首次启动会在 `server/data/` 下建 `app.db`（SQLite —— 配置 + 用户 + 会话 + token 用量全在里面），
并**自动导入**老的 `config.json` + `history.json`（如果还在的话）。

## 配置

### `server/.env`（不进 git）

后端启动时自动加载（`src/config/env.ts`）。`server/.env.example` 是模板。

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `8787` | Web 端口 |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | 仅首次播种默认 provider 时用 |
| `OLLAMA_MODEL` | `s2021008840/hy-mt2:1.8b` | 同上 |
| `MAX_CANDIDATES` | `5` | 候选词数量上限 |
| `<PROVIDER_ID>_KEY` | — | **API Key**，名字对应 provider id（大写 + `-`→`_` + `_KEY`） |
| `VOCAB_PATH` | `.../024-cet4/vocab_all.json` | **查词**用的词库 JSON（JSONL） |
| `DB_PATH` | `server/data/app.db` | SQLite 库文件（配置 + 用户 + 会话 + token 用量） |
| `ADMIN_USERNAME` | `admin` | 首次启动播种的账号名 |
| `ADMIN_PASSWORD` | `admin123` | 同上的密码（**只在播种时用一次**；日志里不回显） |
| `AUTH_LOCAL_BYPASS` | `1` | 本地工具（curl / CLI）免登录放行；设 0 = 全部要登录 |
| `DEBUG` | — | 设 `1` 打开调试日志（server 端 + 浏览器 console 共用一个开关） |

改 `VOCAB_PATH` 不用重启后端 —— 设置页「词库」tab 点「重新加载」即可（后端会重读 `.env` 并重建索引）。
用 `SERVER_ENV_FILE=/path/to/other.env` 可以换一个 env 文件。

### `server/data/`（`*.db` 不进 git）

**所有持久化都在 `server/data/app.db`（SQLite）里** —— 用 Node 内置的 `node:sqlite`，零第三方依赖。

| 表 | 内容 |
|---|---|
| `providers` | provider 列表（id / 顺序 / 配置 JSON） |
| `app_state` | 当前生效的 provider id 等全局状态 |
| `prompt_templates` | 每个 provider 的 6 个能力模板 |
| `prompt_tones` | 每个 provider 的语气表 |
| `prompt_versions` | 提示词的 append-only 版本历史 |
| `users` | 用户（id / 用户名 / 密码哈希 / 建号时间） |
| `sessions` | 登录会话（token / user_id / 过期时间） |
| `token_usage` | **每人每天**的 token 消耗（input / output / 调用次数） |

> `server/data/config.json` 和 `history.json` 现在只是**首次导入源**：
> 库里还没有 provider 时会自动把它们导进来（导入后不再回写）。老文件仍在 git 里，
> 所以全新 clone 出来的库也有你那份 provider 配置。

**库文件不进 git**（`server/.gitignore` 里有 `data/*.db*`）—— 在 UI 里改的提示词只存在本地库里。

`app.db` 里**永远没有明文 key** —— key 只在 `.env` 里。provider 的 `apiKeyEnv` 字段指明读哪个变量名，
API 返回 provider 时只带一个 `hasKey: boolean`。

> 以前 key 存在 `server/data/secrets.json`，那个文件已经删掉了，机制换成 `.env`。

### provider 配置

```json
{
  "id": "openai-main",
  "label": "OpenAI",
  "kind": "openai",
  "baseUrl": "https://api.openai.com/v1",
  "model": "gpt-4o-mini",
  "apiKeyEnv": "OPENAI_API_KEY",
  "structuredOutput": "schema",
  "maxTokens": 4096,
  "maxTokensParam": "max_tokens",
  "timeoutMs": 120000
}
```

| 字段 | 说明 |
|---|---|
| `kind` | `ollama` / `openai` / `anthropic` —— 决定用哪个 SDK、baseUrl 怎么映射 |
| `baseUrl` | Ollama → `host`；OpenAI / Anthropic → `baseURL` |
| `apiKeyEnv` | 从哪个环境变量读 key（`OPENAI_API_KEY` 这种）。留空 = 不用 key |
| `structuredOutput` | `schema`（API 层强制 JSON Schema）/ `object`（只强制 JSON 对象）/ `prompt`（只靠提示词）/ `off` |
| `maxTokensParam` | `max_tokens`（兼容面最广，默认）或 `max_completion_tokens`（OpenAI 新写法） |
| `temperature` | 只在显式配置时才发。**Anthropic 无效** —— 它的 API 已移除采样参数 |

### 提示词模板语法

模板是一段 mustache-lite（`src/config/template.ts`，约 130 行，不引第三方依赖）：

| 写法 | 含义 |
|---|---|
| `{{name}}` | 变量替换；缺失 → 空字符串 |
| `{{#name}}…{{/name}}` | 变量为真值时渲染整块（非空字符串 / 非 false / 非 0） |
| `{{^name}}…{{/name}}` | 变量为假值时渲染整块 |
| `{{! … }}` | 注释，忽略 |
| `{{a.b}}` | 点号路径 |

**standalone 行处理**：section 标签独占一行时，只把这一行里除标签以外的内容去掉（标签前的缩进 + 标签后的换行）。所以

```
Rules:
{{#kindWord}}
- one
{{/kindWord}}
- two
```

渲染成 `Rules:\n- one\n- two` —— 不会多出空行。这条必须做对：提示词是从旧代码逐字迁移过来的，渲染结果要和老 builder **逐字节相同**。

### 各能力可用的变量

| 能力 | 变量 |
|---|---|
| `translate` | `text`、`tone` |
| `translateReverse` | `text`、`tone` |
| `rewriteSelection` | `text`、`context`、`sourceSentence`、`hasSource`、`kindWord`、`kindPhrase` |
| `rewriteSentence` | `sentence`、`operation`、`operationGuidance`、`tone`、`context`、`sourceSentence`、`hasSource` |
| `grammarCheck` | `sentence` |
| `rewriteInstruction` | `text`、`instruction`、`context` |

`hasSource` / `kindWord` / `kindPhrase` 是 section 的开关变量（代码算好塞进来的）；`tone` 装的是**整行**语气指令（`Write in a formal tone: …`），没有语气时是空 → `{{#tone}}` 整块跳过。

### 版本历史与回滚

保存一次提示词 = 追加一条版本记录（`prompt_versions` 表，主键是 `<providerId>/<capabilityId>/<version>`）。

**回滚不改历史** —— 它把旧版本的内容复制成一个**新版本**再设为当前。这样版本号只增不减，回滚本身也是一条可追溯的记录。

## LLM 抽象层

`src/llm/` 下三家客户端实现同一个接口：

```ts
interface LlmClient {
  readonly kind: ProviderKind
  chat(req: { system: string; user: string; jsonSchema?: JsonSchema }): Promise<ChatResult>
  listModels(): Promise<string[]>
}
```

### 结构化输出与降级链

`responseMode: 'json'` 时提示词可以要求模型返回格式化 JSON，各 provider 的强制方式不同：

| provider | 强制方式 |
|---|---|
| Ollama | `format: <JSON Schema 对象>` —— Ollama 原生接受 schema 对象做语法约束 |
| OpenAI | `response_format: { type: 'json_schema', json_schema: { name, schema, strict: false } }` |
| Anthropic | `tools: [{ name: 'result', input_schema: <schema> }]` + `tool_choice: { type: 'tool', name: 'result' }` |

**降级链**（只有 400 才降级，其它错误直接抛）：

- OpenAI：`json_schema` → `json_object` → 完全不带 `response_format`
- Anthropic：带 tools → 不带 tools

降级后还能拿到 JSON，靠的是：**只要请求带了 `jsonSchema`，且 API 层没有在强制，就在 system 后面追加一段标准 JSON 契约**（「Respond with a single JSON object that matches the following JSON Schema…」）。API 层强制时则不加 —— 模型看到的和编辑器里写的完全一致。

### 几个 SDK 坑（都已在代码里处理）

- **有些模型会把推理塞进正文里**。MiniMax-M3 走 **OpenAI 兼容端点**时，`<think>…</think>` 直接混在 `content` 里；走 Anthropic 兼容端点则不会（它单独返回一个 thinking block）。不剥的话后果很具体：语法检查（原文透传）会把整段思考显示给用户，翻译 / 改写的解析会直接失败，选区候选的 JSON 数组也会被思考里的方括号带偏。所以三个客户端都会在返回前调 `stripReasoning()` 剥掉推理块 —— 没有推理块时**原样返回**，不影响「模型写什么就透传什么」的语义。
- **Anthropic 的 `temperature` 已经没有了** —— SDK v1 的 `MessageCreateParams` 上根本没这个字段，API 侧已移除。配置里的 `temperature` 对 Anthropic 无效。
- **Anthropic 的 `max_tokens` 是必填的**，而且它是「思考 + 回复」的共同上限（Opus 5 默认开 adaptive thinking），给小了会截断 → 默认 16000。
- **OpenAI v7 里 `max_tokens` 已标记 deprecated**，但 DeepSeek / Moonshot 这类兼容端点只认它 → 所以做成可配的 `maxTokensParam`，默认用 `max_tokens`。
- **OpenAI 的 `strict: true` 只支持 JSON Schema 子集**（属性全 required + `additionalProperties: false`），用户手写的 schema 很容易不满足 → 固定用 `strict: false`。
- **Anthropic 的 `output_config.format` 只在部分新模型上可用**，且限制更多 → 统一用 tool-use，全模型可用。

### 预置的 provider

库里预置了三个（老 `config.json` 里也是这三个，首次启动会整个导进来）：

| id | 类型 | 端点 | 模型 |
|---|---|---|---|
| `ollama-local` | Ollama | `http://127.0.0.1:11434` | `s2021008840/hy-mt2:1.8b`（本地默认） |
| `minimax-anthropic` | Anthropic | `https://api.minimaxi.com/anthropic` | `MiniMax-M3` |
| `minimax-openai` | OpenAI 兼容 | `https://api.minimaxi.com/v1` | `MiniMax-M3` |

后两个的 key 在 `server/.env` 里（不进 git），变量名分别是 `MINIMAX_ANTHROPIC_KEY` / `MINIMAX_OPENAI_KEY`
（provider 的 `apiKeyEnv` 字段指明读哪个）。两个都开了 `structuredOutput: 'schema'`，可以在设置页里把任意一个能力的提示词切到 JSON 模式。

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
pnpm cli translate --to zh --text "Education is very important for children."   # 英译中
pnpm cli models
```

CLI 走的是和 server 一样的配置（同一个 `app.db` 里的当前生效 provider + 该 provider 的提示词 + `server/.env`），`--provider <id>` / `--model <name>` 可以临时覆盖。

**CLI 不计入任何用户的 token 用量** —— 它不经过 HTTP，没有登录用户身份。

调试用的开关：

| 开关 | 作用 |
|---|---|
| `--prompt` | 打印**实际发出**的 system / user 消息（不是重新拼的，是录到的真实请求） |
| `--raw` | 打印模型原始输出，绕过 `parseCandidates` / `cleanText`——解析失败时用它看是哪一步炸的 |
| `--json` | 把 command / provider / model / prompt / raw / result / error / 耗时一次性输出成 JSON |

## 环境变量

**全部在 `server/.env` 里**（不进 git），启动时自动加载 —— 细节见上面的「配置」一节。
`PORT` / `OLLAMA_HOST` / `OLLAMA_MODEL` / `MAX_CANDIDATES` / `<PROVIDER_ID>_KEY` / `VOCAB_PATH` /
`DB_PATH` / `ADMIN_USERNAME` / `ADMIN_PASSWORD` / `AUTH_LOCAL_BYPASS` / `DEBUG`。

> `SERVER_ENV_FILE=/path/to/other.env` 可以换一个 env 文件。

## 语气

`neutral`（中性）/ `formal`（正式）/ `casual`（口语）/ `professional`（专业）/ `friendly`（友好）/ `concise`（简洁）/ `detailed`（详细）/ `persuasive`（有说服力）。

语气 **id 固定**（代码里 `src/config/defaults.ts` 的 `TONE_IDS` 是唯一来源），但**每个 provider 的 label 和 guidance 各自独立** —— guidance 是要拼进 prompt 正文的，给 1.8B 小模型的应该是「必须是完整句子，不要省略主语」这种直白规则，给 Claude 的应该是「保持克制，不要堆砌形容词」这种松弛指令。

`tone` 是**可选**字段：不传时走各自的默认 prompt，输出与加语气之前逐字节一致；传了才在 system prompt 末尾追加一行 `Write in a <tone> tone: <guidance>.`。传了枚举外的值 → `400`。

## API

### AI 能力

`POST /api/translate` · `POST /api/rewrite-selection` · `POST /api/rewrite-sentence` · `POST /api/grammar-check` · `POST /api/rewrite-instruction` · `GET /healthz`

请求/响应形状完全没变 —— 无论提示词是纯文本模式还是 JSON 模式，出来的形状都一样（翻译 → `{translation}`，选区 → `{candidates}`，其余 → `{text}`）。

**`/api/translate` 现在支持两个方向**：`{text, source?, target?, tone?}`，
默认 `source='zh'` / `target='en'`；`source='en'` + `target='zh'` 走**英译中**（另一个能力 / 另一套提示词）。
`source` / `target` 相同或不是 zh/en → `400`。响应形状不变，还是 `{translation}`。

错误码：

| 状态码 | 含义 |
|---|---|
| `400` | 入参不合法 |
| `401` | 需要登录（见下面的「登录与 token 统计」） |
| `404` | 找不到指定的 provider / 版本 |
| `500` | 模型返回了无法解析的内容 |
| `502` | 上游（provider）调用失败 |
| `503` | 服务端配置不全（没有可用 provider、缺提示词模板） |

### 登录

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/auth/register` | `{username, password}` → 建号 + 直接发 cookie 登录（201） |
| POST | `/api/auth/login` | `{username, password}` → 发 cookie；密码错 → `401` |
| POST | `/api/auth/logout` | 删会话 + 清 cookie |
| GET | `/api/auth/me` | `{user, today, month}` —— 当前用户 + 今日/本月 token 用量；没登录 → `401` |

### 调试

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/debug` | `{enabled}` —— 后端的 `DEBUG` 开关（前端拿它决定要不要打 console.log） |

### 查词（纯本地，不调模型）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/dictionary/status` | `{path, count, loaded, error?}` —— 当前词库路径 + 索引状态 |
| POST | `/api/dictionary/reload` | 重读 `.env` + 重建索引，返回新的 status |
| GET | `/api/dictionary/lookup?word=xxx` | `{found, word, entry?}` —— 词条：音标 / 释义 / 例句 / 短语 / 同反义词 / 记忆法 |

**「查不到」是 200 + `found: false`**，不是 404 —— 查不到是正常结果。
`word` 缺失 → `400`。词库文件没加载起来时 `found:false` 且带 `error` 说明原因。

### 设置

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/config` | 一次性拿到生效 provider、provider 列表、语气表、能力列表 |
| GET | `/api/providers` | provider 列表（带 `hasKey`，没有明文 key） |
| PUT | `/api/providers` | 新增 / 更新。**不收 `apiKey`**（key 在 `.env` 里）—— 带了会 400 |
| DELETE | `/api/providers/:id` | 删除（连带清掉它的提示词和历史）；当前生效的不给删 |
| POST | `/api/providers/active` | 切换生效 provider —— **即时生效，不用重启** |
| POST | `/api/providers/:id/test` | 探活：发一次最小请求 |
| GET | `/api/providers/:id/models` | 拉模型列表 |
| GET | `/api/providers/:id/prompts` | 该 provider 的整套提示词 + 语气表 + 变量表 + 示例输入 |
| GET | `/api/providers/:id/prompts/:capabilityId` | 单个模板 + 版本列表 |
| PUT | `/api/providers/:id/prompts/:capabilityId` | 保存 → 产生新版本 |
| GET | `/api/providers/:id/prompts/:capabilityId/versions` | 版本列表 |
| GET | `/api/providers/:id/prompts/:capabilityId/versions/:v` | 某个版本的完整快照 |
| POST | `/api/providers/:id/prompts/:capabilityId/rollback` | 回滚（复制成新版本） |
| POST | `/api/providers/:id/prompts/:capabilityId/dry-run` | 试运行：渲染 + 真跑一次，返回 system / user / 原始输出 / 解析结果 |
| POST | `/api/providers/:id/prompts/copy` | 从别的 provider 复制提示词（整套或单个能力） |
| GET / PUT | `/api/providers/:id/tones` | 该 provider 的语气表 |

## 登录与 token 统计

### 谁能调哪些接口

**所有消耗 token 的接口都必须登录**（无 session → `401`）：

```
/api/translate
/api/rewrite-selection
/api/rewrite-sentence
/api/grammar-check
/api/rewrite-instruction
/api/providers/:id/test
/api/providers/:id/prompts/:capabilityId/dry-run
```

不花 token 的（`/api/config`、provider 的增删改、查词、`/api/auth/*`）不需要登录。

### 本地工具免登录

浏览器打前端（Vite 5173）→ Vite 代理到后端（8787）→ **后端看到的 IP 也是 127.0.0.1**。
所以只看 IP 会把浏览器流量一起放行 —— 登录在开发环境就等于没有。

区分靠**浏览器一定会带的请求头**：

| | 浏览器（fetch） | curl / CLI |
|---|---|---|
| `Origin` | POST 一定有（`http://127.0.0.1:5173`） | 无 |
| `Sec-Fetch-Site` | 一定有（`same-origin`） | 无 |

规则：**回环地址 + 没有 `Origin` + 没有 `Sec-Fetch-Site` → 判定为本地工具调用，放行**。
所以 `server/curl-tests.sh` 不用改；但这类请求**没有用户身份，不计入任何人的用量**。

`.env` 里 `AUTH_LOCAL_BYPASS=0` 可以把这个放行整个关掉（所有请求都必须登录）。

### 密码和会话

- **密码**：`node:crypto` 的 `scryptSync(password, salt, 64)`，库里存 `saltHex:hashHex`，
  校验用 `timingSafeEqual`（防时序攻击）。零第三方依赖。
- **会话**：token 是 32 字节随机数（base64url），服务端存 `sessions` 表，浏览器只拿 cookie
  （`sid`，HttpOnly / Path=/ / SameSite=Lax / 30 天）。
- 用户名规则：3-32 位 `[A-Za-z0-9_-]`；密码 ≥ 4 位。
- 播种：库里一个用户都没有时，用 `ADMIN_USERNAME` / `ADMIN_PASSWORD` 建一个（**日志里不回显密码**）。

### token 用量怎么记

1. 三个 LLM 客户端把 provider 返回的 usage 塞进 `ChatResult.usage`：
   - Ollama：`prompt_eval_count` / `eval_count`
   - OpenAI：`usage.prompt_tokens` / `usage.completion_tokens`（写在内部 `call()` 里 → 降级链里成功的那次也带得上）
   - Anthropic：`usage.input_tokens` / `usage.output_tokens`（**加上** cache 读写那两段 —— 那也算真实消耗；两个 return 点都要加）
2. service 层（`services/ai.ts` / `services/translate.ts`）在 `chat()` 成功后、`extractOutput()` **之前**记账 ——
   解析失败也算钱，token 已经花了。
3. 「记到谁头上」从 **AsyncLocalStorage** 里取当前用户（`services/requestContext.ts`）。
   登录守卫里 `runAsUser(user, next)` 把身份挂进请求上下文 —— 并发安全，
   而且**不用改任何 service 的方法签名**。
4. 落库：`token_usage` 表每人每天一行（`PRIMARY KEY (user_id, day)`），UPSERT 累加。

**`day` 用 INTEGER（`YYYYMMDD`，如 `20261031`）** —— SQLite 没有真正的 DATE 类型
（声明 `DATE` 只是给 NUMERIC affinity，底层还是文本），而这一列只做等值和范围比较，
INTEGER 比较最干净、没有 affinity 歧义。

日/月边界按**服务端本地时区**算（`getFullYear/getMonth/getDate`），不用 UTC —— 避免东八区晚上记到「明天」。
本月区间是 `[当月1号, 下月1号)` 的开区间，跨年不用特判。

**CLI 不计入任何用户** —— 它不经过 HTTP，没有用户身份。

### 调试日志

`.env` 里 `DEBUG=1` 打开。前后端共用这一个开关（前端通过 `GET /api/debug` 拿）。

server 端会打：启动信息、库路径、登录/登出/401/本地放行、每次 LLM 调用的
（能力 / 模型 / input / output token / 耗时）、token 落库、配置变更。

浏览器端会给 `window.fetch` 打补丁，每个请求一行 `[debug] GET /api/... → 200 (12ms)`。

## 查词服务

`src/services/dictionary.ts` 读的是 **JSONL**（一行一个词条）的本地词库：

```
{"wordRank":1,"headWord":"abandon","content":{"word":{"wordHead":"abandon","content":{
  "usphone":"ə'bændən","ukphone":"ə'bænd(ə)n",
  "trans":[{"tranCn":"丢弃；放弃，抛弃","pos":"v","tranOther":"to leave someone..."}],
  "sentence":{"sentences":[{"sContent":"How could she abandon her own child?","sCn":"她怎么能抛弃自己的孩子？"}]},
  "phrase":{"phrases":[{"pContent":"abandon ship","pCn":"弃船"}]}, ...}}}}
```

**索引用「字节偏移索引」** —— 建索引时把整个文件读成一个 Buffer，按 `\n` 扫，
每行只 `JSON.parse` 一次取 `wordHead`，记下偏移后**立刻丢掉解析结果**。
12MB / 5011 行的词库建索引 ~50ms，内存里只留 5011 条 `Map<string, number>`；
查词 = `seek` 到那一行 → `readline` → parse 这一行（~2.5KB），毫秒级。
（思路和 024-cet4 的 `scripts/cet4.py` 一样。）

词库**只收词头**（`child`），不收变形（`children`）。查不到时会退一步试试常见变形还原
（`-s` / `-es` / `-ed` / `-ing`），命中就返回，并告诉前端「词库里收的是『child』」。

## 设计约定

## 设计约定

**分层**：`config/`（配置 + 提示词模板 + 渲染）→ `llm/`（provider 客户端）→ `services/`（能力编排）→ `routes/`（校验 + 状态码分类）。

**提示词全部可配**，但**解析逻辑仍然是代码** —— `parseCandidates` / `cleanText` 那套降级链留在 `src/prompts/` 里，因为「模型回了段带围栏的 JSON」这种脏活不是配置能表达的。`config/capabilities.ts` 里的 `extractOutput` 负责按模板声明的输出模式（纯文本 / JSON）把模型原始输出转成最终值。

**system 和 user 分开**：语气、操作类型、选区类型这类硬约束走 system message（模型对 system 的服从度更高），不要塞进 user message 再配一个泛泛的 system。

`src/prompts/cleanText.ts` 是翻译 / 整句改写 / 自定义指令三个纯文本能力的共享清洗器：剥 markdown 围栏 → 剥 `翻译：`/`Rewritten:` 这类前缀 → 剥首尾成对引号 → 拒绝空串。**语法检查不过这个清洗器** —— 它的输出就是给用户看的原文，清洗只会改掉模型想说的东西。

`parseCandidates` 用候选降级：整段 JSON → ``` 围栏里的 JSON → 正文里的 `[...]`。关键是**保留第一个「能 parse 但形状不对」的错误**（`shapeError`），只在所有候选都失败时才抛它。

`src/routes/httpError.ts` 里的 `classifyError(err)` 统一做状态码分类（500 解析失败 / 502 上游失败 / 503 配置不全 / 404 不存在 / 400 请求不合法）。

**DI**：`createApp({ aiService, translateService, configStore, registry, dictionaryService, userStore, recordUsage })`。测试注入 mock service，不需要启动真模型。**`userStore` 不传 = 不启用鉴权**（单元测试靠这个绕开登录；生产入口 `src/index.ts` 一定传）。

## 测试

```bash
pnpm test            # 一次性
pnpm test:watch      # watch
pnpm type-check      # tsc --noEmit
```

测试用 `supertest` + `vitest`，通过 `tests/helpers.ts` 注入 mock 打到真实路由层，无需启动真模型。

| 文件 | 覆盖 |
|---|---|
| `tests/promptParity.test.ts` | **迁移安全网**：默认提示词集的渲染结果必须和迁移前的 builder 逐字节相同 |
| `tests/legacy/prompts/*` | 迁移前的原始实现，从 git 抽出来冻存在这儿当对照（**不要改**） |
| `tests/template.test.ts` | 渲染器：变量 / section / inverted / standalone 行裁剪 / 配平检查 |
| `tests/configStore.test.ts` | 播种、per-provider 提示词隔离、版本递增、回滚、key 走环境变量 |
| `tests/llmClients.test.ts` | 三家客户端的请求形状 + 降级链（SDK 用 mock 挡掉） |
| `tests/capabilities.test.ts` | 提取器（text / json 两种模式）、buildContext、renderPrompt |
| `tests/settingsRoutes.test.ts` | 设置 API 全流程 |
| `tests/services.test.ts` | 断言提示词真的进了请求，语法检查原样返回 |
| `tests/rewriteRoutes.test.ts` | 四个端点的 400 / 500 / 502 路径 |
| `tests/tones.test.ts` · `tests/translateWord.test.ts` · `tests/rewriteSelection.test.ts` · `tests/rewriteSentence.test.ts` · `tests/rewriteInstruction.test.ts` · `tests/grammarCheck.test.ts` | 各能力的模板渲染 + 解析器 |
| `tests/translate.test.ts` · `tests/matchCase.test.ts` | 路由层（两个翻译方向）+ 大小写保留 |
| `tests/env.test.ts` | `.env` 加载：读得到 / 覆盖已有变量 / 文件缺失不炸 |
| `tests/dictionary.test.ts` | 查词：索引 + 查询 + 查不到 + 坏行跳过 + 词形还原 + 路由层 |

> **`tests/promptParity.test.ts` 为什么用真代码当对照**：开发这个功能时把 `"do not include it in the output"` 抄成了 `"... in your output"`，结果「模板」和「测试里手抄的期望值」一起错，测试反而是绿的 —— 绿得毫无意义。所以期望值不手抄，直接从 git 抽一份原始实现冻在 `tests/legacy/prompts/` 里。

## 目录

```
server/
├── src/
│   ├── index.ts                        # 入口：loadEnv → 建库 / store / registry / service，播种 admin，listen
│   ├── app.ts                          # createApp 工厂，挂 rewrite / translate / settings / dictionary / auth router
│   ├── cli.ts                          # 手工测试 CLI（直接调 service，可打印 prompt / raw）
│   ├── config/
│   │   ├── types.ts                    # ProviderConfig / PromptTemplate / ProviderPromptSet / AppConfig
│   │   ├── defaults.ts                 # 内置默认提示词集（从旧 prompts/*.ts 逐字迁移）
│   │   ├── template.ts                 # mustache-lite 渲染器
│   │   ├── capabilities.ts             # 能力注册表：变量表 / 示例输入 / 提取器
│   │   ├── env.ts                      # 加载 server/.env（util.parseEnv）
│   │   ├── db.ts                       # openDatabase：建库 + 全部建表语句（node:sqlite，零依赖）
│   │   ├── store.ts                    # ConfigStore：配置存在 SQLite（老 config.json 只作首次导入源）
│   │   └── userStore.ts                # UserStore：用户 / 会话 / 每人每日 token 用量
│   ├── llm/
│   │   ├── types.ts                    # LlmClient 接口 + JSON 契约 + TokenUsage
│   │   ├── ollamaClient.ts             # format 传 schema
│   │   ├── openaiClient.ts             # response_format + 三级降级
│   │   ├── anthropicClient.ts          # tool-use 强制 + 降级
│   │   └── registry.ts                 # ProviderConfig → LlmClient（带缓存）
│   ├── services/
│   │   ├── ai.ts                       # AiService：4 个能力，提示词从 provider 的提示词集取
│   │   ├── translate.ts                # TranslateService（中译英 / 英译中）
│   │   ├── dictionary.ts               # 查词：字节偏移索引 + 本地 JSONL 词库
│   │   ├── activeProvider.ts           # 把 store + registry 拼成「当前生效 provider」快照
│   │   └── requestContext.ts           # AsyncLocalStorage：当前请求的登录用户
│   ├── routes/
│   │   ├── translate.ts                # POST /api/translate（两个方向）
│   │   ├── rewrite.ts                  # 4 个端点：selection / sentence / grammar / instruction
│   │   ├── settings.ts                 # provider / 提示词 / 版本 / 回滚 / 试运行
│   │   ├── dictionary.ts               # /api/dictionary/{status,reload,lookup}
│   │   ├── auth.ts                     # /api/auth/{register,login,logout,me} + requireUser 守卫
│   │   └── httpError.ts                # 统一状态码分类
│   ├── prompts/                        # 只剩解析器（提示词文本已搬到 config/）
│   │   ├── cleanText.ts                # 共享清洗：围栏 / 前缀 / 引号
│   │   ├── translate.ts                # parseTranslation
│   │   ├── rewriteSelection.ts         # isSelectionKind / parseCandidates
│   │   ├── rewriteSentence.ts          # isSentenceOperation / parseRewrittenSentence
│   │   └── rewriteInstruction.ts       # parseInstructionResult
│   └── utils/
│       ├── matchCase.ts                # Good→Great / good→great / GOOD→GREAT
│       ├── day.ts                      # 日桶键（YYYYMMDD 整数）+ 当月区间（本地时区）
│       └── debug.ts                    # debugLog：server/.env 的 DEBUG 开关
├── .env                                 # 本机配置：端口 / Ollama / API Key / 词库路径 / 登录 / 调试（不进 git）
├── data/                               # app.db（不进 git）+ config.json / history.json（老导入源，进 git）
└── tests/                              # vitest
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
bash curl-tests.sh translate-reverse  # /api/translate（英译中）
bash curl-tests.sh dictionary   # /api/dictionary/{status,lookup}
bash curl-tests.sh errors       # 4 个端点的错误路径（11 个 case）
BASE=http://host:port bash curl-tests.sh   # 改 base URL
```

> curl 走的是「本地工具调用」路径 —— 免登录放行（`AUTH_LOCAL_BYPASS=1` 时），
> 所以这个脚本不用改。但这些调用**不计入任何用户的 token 用量**。
> `.env` 里把 `AUTH_LOCAL_BYPASS` 设成 0 的话，curl 也会拿到 401。
