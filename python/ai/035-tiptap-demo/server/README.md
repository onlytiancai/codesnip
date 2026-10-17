# tiptap-rewrite-server

Tiptap 编辑器「单词替换」功能的后端。包装本地 [Ollama](https://ollama.com/) 模型，把选中单词和上下文发给模型，解析回 JSON 候选词列表。

## 启动

```bash
cp .env.example .env       # 按需改 OLLAMA_MODEL
pnpm install
pnpm dev                   # tsx watch，监听 8787
```

可用模型示例：`s2021008840/hy-mt2:1.8b`（默认）。前提：本地 Ollama 已启动并已 `ollama pull` 该模型。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `OLLAMA_MODEL` | `s2021008840/hy-mt2:1.8b` | Ollama 模型名 |
| `PORT` | `8787` | Web 端口 |
| `MAX_CANDIDATES` | `5` | 候选上限（实际取模型返回值与 max 较小者） |

> `OLLAMA_HOST` 不读取——直接走 `ollama` npm 包的默认值（`http://127.0.0.1:11434`）。要改请在 `src/services/ollama.ts` 里注入 client。

## API

### `POST /api/rewrite-word`

请求体：
```json
{ "word": "good", "context": "This is a good idea." }
```

`context` 可省略。

成功响应 `200`：
```json
{ "candidates": ["great", "promising", "excellent"] }
```

错误码：

| 状态码 | 含义 |
|---|---|
| `400` | 缺 `word` / `word` 非字符串 / `context` 非字符串 |
| `500` | 模型返回了无法解析为 JSON 数组的内容 |
| `502` | Ollama 上游调用失败（ECONNREFUSED 等） |

### `GET /healthz`

冒烟用，返回 `{ "ok": true }`。

## 测试

```bash
pnpm test            # 一次性
pnpm test:watch      # watch
pnpm type-check      # tsc --noEmit
```

测试用 `supertest` + `vitest`，通过注入 mock `OllamaService` 打到真实路由层，无需启动真 Ollama。

- `tests/matchCase.test.ts`：大小写保留（5）
- `tests/rewriteWord.test.ts`：prompt 构建 + JSON 解析（15）
- `tests/rewrite.test.ts`：路由层（11）

合计 31 个用例。

## 目录

```
server/
├── src/
│   ├── index.ts                # 入口，listen
│   ├── app.ts                  # createApp 工厂
│   ├── routes/rewrite.ts       # POST /api/rewrite-word
│   ├── services/ollama.ts      # createOllamaService（client 可注入）
│   ├── prompts/rewriteWord.ts  # prompt 模板 + 解析
│   └── utils/matchCase.ts      # Good→Great / good→great / GOOD→GREAT
└── tests/                      # vitest
```