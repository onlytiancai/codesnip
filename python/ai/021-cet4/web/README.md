# 021-cet4 web

CET-4 真题核心词高频词 v3 可视化浏览（Vue 3 + Tailwind 4 + MiniMax TTS）。

按 **25 个分类 · 296 组 · 1162 词** 浏览；点词 chip 打开右侧抽屉看完整字典详情
（音标、释义、同近义词、短语、同根词、例句、真题例句、记忆方法、真题）；点 🔊 让
MiniMax TTS 朗读单词 / 中文讲解。

无构建步骤，单文件启动：

```bash
cd web
pnpm install
cp .env.example .env
# 编辑 .env，把 MINIMAX_API_KEY 填进去
pnpm start
# 浏览 http://localhost:3000
```

---

## 目录

```
web/
├── server.js           ← Express + dotenv + TTS 路由 (ESM)
├── package.json        ← 仅 express + dotenv,无前端构建依赖
├── .env.example        ← TTS 配置模板
├── .gitignore          ← .env + data/tts-cache/
├── data/
│   └── tts-cache/      ← TTS 磁盘缓存 (按 sha1 文件名堆积的 .mp3)
└── public/
    ├── index.html      ← Vue 3 + Tailwind 4 SPA
    ├── app.js          ← Vue 3 Options API (单 createApp)
    ├── vendor-vue.js   ← vendor 化 (cdn.staticfile.net/vue/3.5.22)
    └── vendor-tw.js    ← vendor 化 (unpkg.com/@tailwindcss/browser@4)
```

## 数据源

默认指向 `../`（即 `021-cet4/`）下的两个文件，相对 `web/server.js` 解析：

| 文件 | 作用 |
|---|---|
| `../cet4_hexinci_groups_v3.json` | 预聚合好的 25 分类 / 296 组 / 1162 词 |
| `../CET4luan_1.json`             | NDJSON 词典（每行一条 entry，1162 条） |

启动期一次性预热：v3 JSON 走 `JSON.parse`，CET4luan_1.json 走按行 `JSON.parse`
（NDJSON 不是 JSON 数组，整文件 `JSON.parse` 会爆）。

可通过环境变量覆盖：
- `DATA_GROUPS=../其他.json`
- `DATA_DICT=../其他.json`

## 路由

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/api/health`     | 健康检查（含 MINIMAX_API_KEY 是否配置 + 数据统计） |
| GET  | `/api/categories` | 分类汇总（侧栏徽章用） |
| GET  | `/api/data`       | 完整 v3 JSON（含 `categories[]` 树） |
| GET  | `/api/dict/:word` | 按单词查 NDJSON 词典；404 返回 `{error:'not found'}` |
| POST | `/api/tts`        | 代理 MiniMax TTS，SHA1 磁盘缓存 |
| GET  | `*`               | SPA 兜底 → `public/index.html` |

## TTS 缓存

按 `${MINIMAX_TTS_MODEL}|${kind}|${voice}|${speed}|${emotion}|${text}` 计算 SHA1，
命中 `data/tts-cache/{hash}.mp3` 直接流式回放（`X-TTS-Cache: HIT`），未命中 POST
`https://api.minimaxi.com/v1/t2a_v2` 把 `data.audio`（hex）落盘再返回。

- 上限 8000 字符（适配单 group `explanation_zh` 平均 860 字 / 最大 1381 字）
- 三种 kind：`en`（英文单词，`English_PassionateWarrior`）/ `zh` + `explain`
  （中文讲解共用 `male-qn-qingse`）

清缓存：`rm -rf data/tts-cache/*.mp3`

## 前端

- **Vue 3.5.22** via `vendor-vue.js`（cdn.staticfile.net 离线 vendor）
- **Tailwind 4 browser** via `vendor-tw.js`（unpkg `@tailwindcss/browser@4`）
- **Options API** 单一 `createApp`，管理：
  - 25 分类列表（侧栏 / `<lg` 移动抽屉）
  - 当前选中分类 + group 长卡
  - 同分类卡片网格（`sm:grid-cols-2 xl:grid-cols-3`）
  - 单词详情抽屉（音标 + 8 个 section）
  - TTS 单例 `Audio` + `playingId` 高亮
- **明亮配色**：`bg-gradient-to-br from-sky-50 via-white to-amber-50`，强调色 `amber` /
  `sky` / `emerald`
- **响应式**：`<lg` 隐藏侧栏，提供 hamburger 触发移动分类抽屉；单词详情抽屉在 mobile
  占满 100%
- **a11y**：所有交互元素带 `aria-label`，`aria-pressed` 同步分类选中态，Esc 关闭
  任意抽屉

## 用户操作

| 行为 | 效果 |
|---|---|
| 点左侧分类 | 切到该分类下第一个 group |
| 点 grid 卡片 | 切到那个 group 显示在长卡 |
| 点词 chip 单词 | 打开右侧抽屉，加载 `GET /api/dict/:word` |
| 点词 chip 🔊 | 朗读该英文单词（`kind='en'`） |
| 点长卡「朗读讲解」 | 朗读该 group 的 `explanation_zh`（`kind='zh'`） |
| 抽屉 × / Esc / 背景遮罩 | 关闭抽屉 |
| 移动端 hamburger | 打开移动分类抽屉（左滑入） |

## 验证清单

```bash
# 1. 服务起来后应打印
loaded 296 groups, 25 categories, 1162 dict entries | words 1162
MINIMAX_API_KEY 已设置

# 2. 健康检查
curl -sS http://localhost:3000/api/health
# {"ok":true,"hasKey":true,"groups":296,"categories":25,"words":1162,"dictEntries":1162}

# 3. 分类数量
curl -sS http://localhost:3000/api/categories | jq 'length'  # → 25

# 4. 单词查询
curl -sS http://localhost:3000/api/dict/access | jq '.content.word.content.usphone'

# 5. TTS
curl -sS -X POST http://localhost:3000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello world","kind":"en"}' -o test.mp3 -D -
# 期望 X-TTS-Cache: MISS（首次）；再次同请求 → HIT
```

## 与 019 (en-reader-video) 的关系

完全复用了 019 的「极简 Express + Vue 3 + 磁盘缓存 TTS」架构：

- `server.js` 的 TTS 路由：`/api/tts` POST 路由、`makeCacheKey`、`clamp`、hex → Buffer
  、MiniMax API body 字段定义 — 全部沿用
- `app.js` 的 `playTTS` 单例音频 + `URL.createObjectURL` + `playingId` 高亮 — 思路
  一致
- 仅差异：TTS 上限 `8000` 字（019 默认 `4000`，本题 explanation_zh 平均 860 字；
  8000 更稳）；新增 NDJSON 词典查询（019 无字典源）

## 风险与降级

| 风险 | 现状 |
|---|---|
| `CET4luan_1.json` 是 NDJSON 不是 JSON 数组 | 已按行 `JSON.parse`，启动期打印 dictTotal |
| Tailwind 4 CDN 不稳 | vendor 化 `vendor-tw.js`，离线可跑 |
| `staticfile.org` 404 | 改用 `cdn.staticfile.net`，版本 3.5.22 |
| TTS 超过 8000 字 | 上限 8000；前端可按 `。` 切前 N 段；超出弹 toast |
| v3 JSON 顶层重复 group | Step 0 已删除 `groups[]` 字段并重新分配全局 group_id |
| chrome devtools viewport | 用 `emulate viewport='WxHxDPR'`，resize_page 不稳 |
