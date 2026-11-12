# 019-en-reader-video

英文精读视频生成器 + Web 展示页。一篇 markdown → 句句中文讲解 + TTS 朗读 → 浏览器逐段精读。

## 快速开始

```bash
pnpm install
cp .env.example .env   # 填入 MINIMAX_API_KEY
pnpm gen split 001     # 1. 切段落/句子 (无 LLM)
pnpm gen intro 001     # 2. 生成 intro/outro (LLM)
pnpm gen sentences 001 # 3. 生成逐句翻译+讲解+段尾总结 (LLM,慢)
pnpm start             # 4. 浏览器打开 http://localhost:3000
```

`PROJECT=002 pnpm gen split 002` 或 `pnpm start --project=002` 切换项目。

## 三步生成

| 命令 | 作用 | LLM |
|------|------|-----|
| `pnpm gen split <p> [id]` | 切段落/句子,带输入体检 | — |
| `pnpm gen intro <p>` | 生成中文 intro + outro | ✓ |
| `pnpm gen sentences <p>` | 处理全部段落 | ✓ |
| `pnpm gen sentence <p> p9` | 重新处理单段 (断点续跑) | ✓ |

每句完成后立即落盘,中断可继续。

任意命令后可加 `--debug`,每次 LLM 请求/响应会追加到 `data/generate-debug-<project>.log`。

## 目录

```
projects/<project>/
  index.md            ← 原始 markdown
  explanations.json   ← 生成产物 (前端读取)
data/tts-cache/       ← TTS 磁盘缓存 (SHA1)
public/               ← Vue 3 + Tailwind 4 前端
generate.ts           ← 离线生成脚本
server.js             ← Express 后端 (静态 + /api/tts)
```

## 环境变量

`.env` 里至少要 `MINIMAX_API_KEY`。常用项:

- `MINIMAX_TTS_MODEL` `EN_VOICE_ID` `ZH_VOICE_ID` `TTS_SPEED`
- `MINIMAX_LLM_MODEL` (默认 `MiniMax-M3`)

## 依赖

Node ≥ 18。LLM/TTS 走 MiniMax (`api.minimaxi.com`)。
