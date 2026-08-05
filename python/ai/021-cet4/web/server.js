// 021-cet4 / 极简 Express + Vue 3 应用入口
// 职责:静态托管 + 暴露 v3 groups(已聚类)+ 查 CET4luan_1.json 字典 + 代理 MiniMax TTS 并做磁盘缓存
//
// 用法:
//   pnpm install
//   cp .env.example .env && 编辑填 MINIMAX_API_KEY
//   pnpm start
//
// 数据源:
//   ../cet4_hexinci_groups_v3.json  ← 预聚合好的 25 类 / 296 组 / 1162 词
//   ../CET4luan_1.json              ← NDJSON 词典（每行一条 entry）

import 'dotenv/config'
import express from 'express'
import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PORT = Number(process.env.PORT || 3000)
const CACHE_DIR = path.join(__dirname, 'data', 'tts-cache')
fs.mkdirSync(CACHE_DIR, { recursive: true })

const DATA_GROUPS = path.resolve(__dirname, process.env.DATA_GROUPS || '../cet4_hexinci_groups_v3.json')
const DATA_DICT = path.resolve(__dirname, process.env.DATA_DICT || '../CET4luan_1.json')

// --- 启动期数据预热 ---
// v3 JSON: 直接 JSON.parse，categories 已经是 25 大类的树形结构
const db = JSON.parse(fs.readFileSync(DATA_GROUPS, 'utf8'))
const categories = db.categories
const totalGroups = db.total_groups
const totalWords = db.vocab_size

// 派生:分类汇总（给侧栏徽章用）
const categorySummary = categories.map((c) => ({
  id: c.category_id,
  name_zh: c.name_zh,
  name_en: c.name_en || '',
  groupCount: c.groups.length,
  wordCount: c.groups.reduce((sum, g) => sum + g.words.length, 0),
}))

// CET4luan_1.json: NDJSON 按行解析
const dictLines = fs.readFileSync(DATA_DICT, 'utf8').split('\n')
const dictIndex = new Map()
let dictTotal = 0
for (const line of dictLines) {
  const t = line.trim()
  if (!t) continue
  dictTotal += 1
  const e = JSON.parse(t)
  // headWord 是大小写原样；以小写为 key；首个出现的获胜
  const k = (e.headWord || '').toLowerCase()
  if (k && !dictIndex.has(k)) dictIndex.set(k, e)
}
console.log(
  `loaded ${totalGroups} groups, ${categories.length} categories, ${dictTotal} dict entries` +
    ` | words ${totalWords}`,
)

const app = express()
app.use(express.json({ limit: '1mb' }))

// --- 静态托管 ---
app.use(
  express.static(path.join(__dirname, 'public'), {
    etag: true,
    maxAge: '1h',
  }),
)

// --- MiniMax TTS 配置（可通过 .env 覆盖） ---
const TTS_ENDPOINT = process.env.MINIMAX_TTS_ENDPOINT || 'https://api.minimaxi.com/v1/t2a_v2'
const TTS_MODEL = process.env.MINIMAX_TTS_MODEL || 'speech-02-hd'
const DEFAULT_VOICE = {
  en: process.env.EN_VOICE_ID || 'English_PassionateWarrior',
  zh: process.env.ZH_VOICE_ID || 'Chinese_patitent_teacher',
  explain: process.env.ZH_VOICE_ID || 'Chinese_patitent_teacher',
}
const DEFAULT_SPEED = Number(process.env.TTS_SPEED || 1.0)
const DEFAULT_EMOTION = {
  en: process.env.EN_TTS_EMOTION || 'fluent',
  zh: process.env.ZH_TTS_EMOTION || 'calm',
  explain: process.env.EXPLAIN_TTS_EMOTION || 'calm',
}

// --- 路由:健康检查（也顺手给前端展示 API Key 状态） ---
app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    hasKey: Boolean(process.env.MINIMAX_API_KEY),
    groups: totalGroups,
    categories: categories.length,
    words: totalWords,
    dictEntries: dictTotal,
  })
})

// --- 路由:分类汇总 ---
app.get('/api/categories', (_req, res) => {
  res.set('Content-Type', 'application/json; charset=utf-8')
  res.json(categorySummary)
})

// --- 路由:完整 v3 JSON（含 categories[] 树） ---
app.get('/api/data', (_req, res) => {
  res.set('Content-Type', 'application/json; charset=utf-8')
  res.set('Cache-Control', 'no-store')
  res.sendFile(DATA_GROUPS)
})

// --- 路由:按单词查字典 ---
app.get('/api/dict/:word', (req, res) => {
  // URL 已自动 decode，这里再 toLowerCase
  const word = decodeURIComponent(req.params.word || '').trim().toLowerCase()
  if (!word) return res.status(400).json({ error: 'word is required' })
  const entry = dictIndex.get(word)
  if (!entry) return res.status(404).json({ error: 'not found', word })
  res.set('Content-Type', 'application/json; charset=utf-8')
  res.json(entry)
})

// --- 工具:clamp + make cache key ---
function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n))
}
function makeCacheKey({ kind, voice, speed, emotion, text }) {
  return crypto
    .createHash('sha1')
    .update(`${TTS_MODEL}|${kind}|${voice}|${speed}|${emotion}|${text}`)
    .digest('hex')
}

// --- 路由:POST /api/tts ---
// body: { text: string, kind?: 'en'|'zh'|'explain', speed?: number }
// → 返回 mp3 二进制;命中磁盘缓存零开销命中
app.post('/api/tts', async (req, res) => {
  const { text, kind = 'en' } = req.body || {}
  const speed = clamp(Number(req.body?.speed ?? DEFAULT_SPEED), 0.5, 2.0)

  if (typeof text !== 'string' || text.trim() === '') {
    return res.status(400).json({ error: 'text is required' })
  }
  // 单 group explanation_zh 平均 860 字 / 最大 1381;但严谨起见做更高的上限
  if (text.length > 8000) {
    return res
      .status(413)
      .json({ error: 'text too long (>8000 chars); please truncate client-side' })
  }

  const voice = DEFAULT_VOICE[kind] || DEFAULT_VOICE.zh
  const emotion = DEFAULT_EMOTION[kind] || DEFAULT_EMOTION.zh
  const cacheKey = makeCacheKey({ kind, voice, speed, emotion, text })
  const cachePath = path.join(CACHE_DIR, `${cacheKey}.mp3`)

  if (fs.existsSync(cachePath)) {
    res.set('Content-Type', 'audio/mpeg')
    res.set('Cache-Control', 'public, max-age=86400')
    res.set('X-TTS-Cache', 'HIT')
    return fs.createReadStream(cachePath).pipe(res)
  }

  const apiKey = process.env.MINIMAX_API_KEY
  if (!apiKey) {
    return res.status(500).json({
      error: 'MINIMAX_API_KEY 未设置;请在 .env 里填好后重启服务。',
    })
  }

  const payload = {
    model: TTS_MODEL,
    text,
    stream: false,
    language_boost: kind === 'en' ? 'English' : 'Chinese',
    voice_setting: {
      voice_id: voice,
      speed,
      vol: 1.0,
      pitch: 0,
      emotion,
    },
    audio_setting: {
      sample_rate: 32000,
      bitrate: 128000,
      format: 'mp3',
      channel: 1,
    },
  }

  try {
    const apiRes = await fetch(TTS_ENDPOINT, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })

    if (!apiRes.ok) {
      const errText = await apiRes.text()
      return res.status(apiRes.status).json({
        error: `MiniMax TTS ${apiRes.status} ${apiRes.statusText}`,
        detail: errText.slice(0, 500),
      })
    }

    const body = await apiRes.json()
    if (body?.base_resp?.status_code !== 0) {
      return res.status(500).json({
        error: body?.base_resp?.status_msg || 'TTS base_resp not success',
        raw: body,
      })
    }

    const hex = body?.data?.audio
    if (!hex) return res.status(500).json({ error: 'empty data.audio in TTS response' })

    const audioBuf = Buffer.from(hex, 'hex')
    fs.writeFileSync(cachePath, audioBuf)

    res.set('Content-Type', 'audio/mpeg')
    res.set('Cache-Control', 'public, max-age=86400')
    res.set('X-TTS-Cache', 'MISS')
    res.set('X-Audio-Length-ms', String(body?.extra_info?.audio_length ?? ''))
    res.send(audioBuf)
  } catch (e) {
    res.status(500).json({ error: `TTS 调用异常: ${e.message}` })
  }
})

// --- 兜底:所有未匹配的 GET 返回 index.html (SPA 风) ---
app.get('*', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')))

app.listen(PORT, () => {
  console.log(`\n🟢 021-cet4 web 已启动`)
  console.log(`   → http://localhost:${PORT}`)
  console.log(`   → 数据源: ${path.relative(__dirname, DATA_GROUPS)}`)
  console.log(`   → 字典源: ${path.relative(__dirname, DATA_DICT)}`)
  console.log(`   → TTS endpoint: ${TTS_ENDPOINT}  (model: ${TTS_MODEL})`)
  console.log(
    `   → ${process.env.MINIMAX_API_KEY ? '✅' : '⚠️ '} MINIMAX_API_KEY ${process.env.MINIMAX_API_KEY ? '已设置' : '未设置 (TTS 请求会失败)'}`,
  )
  console.log('')
})
