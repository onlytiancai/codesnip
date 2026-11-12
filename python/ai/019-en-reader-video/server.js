// 019-en-reader-video / 极简 Express + Vue 3 应用入口
// 负责:静态托管 + 暴露 JSON + 代理 MiniMax TTS 并做磁盘缓存
//
// 用法:
//   pnpm install
//   pnpm gen split 001 && pnpm gen intro 001 && pnpm gen sentences 001
//   pnpm start                         # 默认服务项目 001
//   pnpm start --project=002            # 服务项目 002
//   PROJECT=002 node server.js          # 同上,环境变量版
//   PORT=8080 pnpm start                # 改端口

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

// --- 项目识别 (CLI > 环境变量 > 默认 001) ---
function detectProject() {
  // --project=XXX 长选项
  const longArg = process.argv.find((a) => a.startsWith('--project='))
  if (longArg) return longArg.split('=')[1]
  // --project XXX 双段长选项
  const longIdx = process.argv.indexOf('--project')
  if (longIdx > -1 && process.argv[longIdx + 1]) return process.argv[longIdx + 1]
  // 位置参数: node server.js 002
  const positional = process.argv[2]
  if (positional && !positional.startsWith('--')) return positional
  // 环境变量
  if (process.env.PROJECT) return process.env.PROJECT
  // 默认
  return '001'
}
const PROJECT = detectProject()
const DATA_PATH = path.join(__dirname, 'projects', PROJECT, 'explanations.json')

const app = express()
app.use(express.json({ limit: '1mb' }))

// --- 静态托管 ---
app.use(express.static(path.join(__dirname, 'public'), {
  etag: true,
  maxAge: '1h',
}))

// --- MiniMax TTS 配置（通过 .env 或环境变量覆盖） ---
const TTS_ENDPOINT = process.env.MINIMAX_TTS_ENDPOINT || 'https://api.minimaxi.com/v1/t2a_v2'
const TTS_MODEL = process.env.MINIMAX_TTS_MODEL || 'speech-02-hd'
const DEFAULT_VOICE = {
  en: process.env.EN_VOICE_ID || 'English_PassionateWarrior',
  zh: process.env.ZH_VOICE_ID || 'male-qn-qingse',
  explain: process.env.ZH_VOICE_ID || 'male-qn-qingse',
}
const DEFAULT_SPEED = Number(process.env.TTS_SPEED || 1.0)
const DEFAULT_EMOTION = {
  en: process.env.EN_TTS_EMOTION || 'fluent',
  zh: process.env.ZH_TTS_EMOTION || 'calm',
  explain: process.env.EXPLAIN_TTS_EMOTION || 'calm',
}

// --- 路由 1:返回 explanations.json (项目级路径) ---
app.get('/api/explanations', (_req, res) => {
  if (!fs.existsSync(DATA_PATH)) {
    return res.status(404).json({
      error: `项目 ${PROJECT} 的 explanations.json 还没生成`,
      expected: path.relative(__dirname, DATA_PATH),
      hint: `pnpm gen split ${PROJECT} && pnpm gen intro ${PROJECT} && pnpm gen sentences ${PROJECT}`,
    })
  }
  res.set('Content-Type', 'application/json; charset=utf-8')
  res.set('Cache-Control', 'no-store')
  res.sendFile(DATA_PATH)
})

// --- 路由:返回当前项目名 ---
app.get('/api/project', (_req, res) => {
  res.json({
    project: PROJECT,
    dataPath: path.relative(__dirname, DATA_PATH),
    exists: fs.existsSync(DATA_PATH),
  })
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

// --- 路由 2:POST /api/tts ---
// body: { text: string, kind?: 'en'|'zh'|'explain', speed?: number }
// → 直接返回 mp3 二进制;命中磁盘缓存则零开销命中
app.post('/api/tts', async (req, res) => {
  const { text, kind = 'en' } = req.body || {}
  const speed = clamp(Number(req.body?.speed ?? DEFAULT_SPEED), 0.5, 2.0)

  if (typeof text !== 'string' || text.trim() === '') {
    return res.status(400).json({ error: 'text is required' })
  }
  if (text.length > 4000) {
    return res.status(413).json({ error: 'text too long (>4000 chars); please truncate client-side' })
  }

  const voice = DEFAULT_VOICE[kind] || DEFAULT_VOICE.zh
  const emotion = DEFAULT_EMOTION[kind] || DEFAULT_EMOTION.zh
  const cacheKey = makeCacheKey({ kind, voice, speed, emotion, text })
  const cachePath = path.join(CACHE_DIR, `${cacheKey}.mp3`)

  // 命中缓存 → 直接流式回放
  if (fs.existsSync(cachePath)) {
    res.set('Content-Type', 'audio/mpeg')
    res.set('Cache-Control', 'public, max-age=86400')
    res.set('X-TTS-Cache', 'HIT')
    return fs.createReadStream(cachePath).pipe(res)
  }

  const apiKey = process.env.MINIMAX_API_KEY
  if (!apiKey) {
    return res.status(500).json({
      error: 'MINIMAX_API_KEY 未设置。请 `export MINIMAX_API_KEY=...` 后重启服务。',
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

// --- 健康检查 ---
app.get('/api/health', (_req, res) => res.json({ ok: true, ts: Date.now() }))

// --- 兜底:所有未匹配的 GET 返回 index.html (SPA 风,虽然本项目用不上) ---
app.get('*', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')))

app.listen(PORT, () => {
  console.log(`\n🟢 019-en-reader-video web 已启动`)
  console.log(`   → http://localhost:${PORT}`)
  console.log(`   → 当前项目: ${PROJECT}  ${fs.existsSync(DATA_PATH) ? '✅' : '⚠️  explanations.json 不存在'}`)
  console.log(`   → TTS endpoint: ${TTS_ENDPOINT}  (model: ${TTS_MODEL})`)
  console.log(`   → ${process.env.MINIMAX_API_KEY ? '✅' : '⚠️ '} MINIMAX_API_KEY ${process.env.MINIMAX_API_KEY ? '已设置' : '未设置 (TTS 请求会失败)'}`)
  console.log('')
})
