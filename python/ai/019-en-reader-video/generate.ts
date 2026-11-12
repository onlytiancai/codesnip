#!/usr/bin/env tsx
/**
 * generate.ts — 三步生成 explanations.json
 *
 * 用法:
 *   pnpm gen split 001               # 切分段落/句子 (无 LLM)
 *   pnpm gen intro 001               # 生成 intro + outro (LLM)
 *   pnpm gen sentences 001           # 处理全部段落 (translation+explanation+summary)
 *   pnpm gen sentence 001 p9         # 处理单段 (paragraph id, 例 p9)
 *
 * 目录约定:
 *   projects/<project>/index.md           ← 原始 markdown
 *   projects/<project>/explanations.json  ← 输出 (前端通过 server.js /api/explanations 取)
 */

import 'dotenv/config'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { existsSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// ===== CLI =====
const __argv = process.argv.slice(2)
const DEBUG = __argv.includes('--debug')
const _argv = __argv.filter((a) => a !== '--debug')
const CMD = _argv[0]
const PROJECT = _argv[1]
const TARGET = _argv[2]

if (!CMD || !PROJECT) {
  printUsage()
  process.exit(1)
}

const INPUT_MD = path.join(__dirname, 'projects', PROJECT, 'index.md')
const OUTPUT_JSON = path.join(__dirname, 'projects', PROJECT, 'explanations.json')
const LOG_FILE = path.join(__dirname, 'data', `generate-debug-${PROJECT}.log`)

function printUsage() {
  console.log(`用法:

  pnpm gen split 001               # Step 1: 切分段落/句子 (无 LLM)
  pnpm gen intro 001               # Step 2: 生成 intro + outro (LLM)
  pnpm gen sentences 001           # Step 3: 处理全部段落
  pnpm gen sentence 001 p9         # Step 3: 处理单段 (paragraph id, 例 p9)

 任意命令后可加 --debug,会把 LLM 请求/响应写到:
   data/generate-debug-<project>.log
`)
}

// debug 日志的总开关:在 import 时确定,后续任何地方都能用
;(globalThis as any).__GEN_DEBUG__ = DEBUG

// ===== JSON IO =====

let current: any = null

async function readJSON<T = any>(p: string): Promise<T> {
  return JSON.parse(await readFile(p, 'utf-8')) as T
}

async function writeJSON(p: string, data: any): Promise<void> {
  await mkdir(path.dirname(p), { recursive: true })
  await writeFile(p, JSON.stringify(data, null, 2), 'utf-8')
}

async function loadProject(): Promise<any> {
  if (!existsSync(OUTPUT_JSON)) {
    throw new Error(
      `explanations.json 不存在: ${OUTPUT_JSON}\n请先执行: pnpm gen split ${PROJECT}`,
    )
  }
  current = await readJSON(OUTPUT_JSON)
  return current
}

async function saveProject(): Promise<void> {
  if (!current) return
  await writeJSON(OUTPUT_JSON, current)
}

// ===== Markdown 预处理 (无 LLM,纯本地) =====

function stripFrontmatter(t: string): string {
  // Hugo/Jekyll 风格 --- 包裹的 YAML 头
  return t.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, '')
}

function stripMarkdown(t: string): string {
  // 代码块
  t = t.replace(/```[\s\S]*?```/g, '')
  // 行内代码
  t = t.replace(/`([^`]+)`/g, '$1')
  // 粗体
  t = t.replace(/\*\*([^*]+)\*\*/g, '$1')
  t = t.replace(/__([^_]+)__/g, '$1')
  // 斜体 (只对 * 起作用,避免误伤 snake_case)
  t = t.replace(/\*([^*\n]+)\*/g, '$1')
  // 链接: 保留文字,丢掉 URL
  t = t.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
  // 标题前缀
  t = t.replace(/^#+\s*/gm, '')
  // 列表标记 (注意: char class 里没有 空格,否则会把"只有空格的行"吃掉)
  t = t.replace(/^[>\-\*\+]\s+/gm, '')
  // 数字列表 (1. 2. ...)
  t = t.replace(/^\d+\.\s+/gm, '')
  // 分隔线
  t = t.replace(/^[-\*_]{3,}\s*$/gm, '')
  // HTML 标签
  t = t.replace(/<[^>]+>/g, '')
  // ★关键: 去掉每行行尾空格 → Markdown 硬换行 "  \n" 也变 "\n"
  // 这一步之后,"段落之间靠 `\n  \n`" 的 markdown 就会被正则识别为真的空行
  t = t.replace(/[ \t]+(?=\r?\n)/g, '')
  return t
}

/** 用 Intl.Segmenter 切英文句子 (Node 18+ 自带) */
function splitSentencesEN(text: string): string[] {
  const seg = new Intl.Segmenter('en', { granularity: 'sentence' })
  const out: string[] = []
  for (const s of seg.segment(text)) {
    const t = (s.segment ?? '').trim()
    if (t) out.push(t)
  }
  return out
}

// ===== Step 1: split =====

/**
 * 简单体检输入文件:在 split 之前把"一眼就发现的问题"列出来,不抛异常,只 console.warn
 * 真正会让 split 失败的严重错误由调用方在外部判断 (例如文件不存在/为空)
 */
function validateInput(raw: string, filePath: string): void {
  const issues: string[] = []
  const warnings: string[] = []

  // 1. 空白文件
  if (raw.trim() === '') {
    issues.push('文件内容为空 (只有空白字符)')
  }

  // 2. 文件大小
  const bytes = Buffer.byteLength(raw, 'utf-8')
  if (bytes < 200) {
    warnings.push(`文件很小 (${bytes} 字节),内容可能不完整`)
  }
  if (bytes > 500_000) {
    warnings.push(`文件很大 (${(bytes / 1024).toFixed(1)} KB),split 之后段落/句子数可能很多`)
  }

  // 3. BOM
  if (raw.charCodeAt(0) === 0xfeff) {
    warnings.push('文件开头含 UTF-8 BOM,会被自动去除')
  }

  // 4. 行级统计
  const lines = raw.split(/\r?\n/)
  const nonEmpty = lines.filter((l) => l.trim() !== '').length
  const blankRatio = lines.length === 0 ? 0 : (lines.length - nonEmpty) / lines.length

  // 4a. 整篇只剩 1 行 → 几乎肯定是换行被吃掉了
  if (lines.length === 1 && raw.length > 500) {
    issues.push(`整篇只有 1 行 (${raw.length} 字符),疑似换行符丢失,会导致 split 出 1 个段落`)
  }

  // 4b. 超长行
  const longLine = lines.findIndex((l) => l.length > 5000)
  if (longLine >= 0) {
    warnings.push(`第 ${longLine + 1} 行超过 5000 字符 (${lines[longLine].length}),可能被错误合并`)
  }

  // 4c. 大量连续空行 (>=3 表示可能格式异常)
  if (/(?:\r?\n){4,}/.test(raw)) {
    const m = raw.match(/(?:\r?\n){4,}/)
    warnings.push(`发现 ${m![0].split('\n').length - 1} 个连续空行,可能多了一个空行`)
  }

  // 4d. 空行比例
  if (lines.length > 10 && blankRatio > 0.5) {
    warnings.push(`空行占比 ${(blankRatio * 100).toFixed(0)}% 偏高`)
  }

  // 5. frontmatter (--- 开头)
  if (/^---\r?\n/.test(raw)) {
    warnings.push('检测到 YAML frontmatter,会被自动剥离')
  }

  // 6. 含非英文/中文异常字符 (控制字符、替换字符 等)
  const ctrlChars = raw.match(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g)
  if (ctrlChars && ctrlChars.length > 0) {
    warnings.push(`发现 ${ctrlChars.length} 个控制字符 (0x00-0x1F),可能 copy-paste 带入`)
  }
  if (/\ufffd/.test(raw)) {
    warnings.push('发现 Unicode 替换字符 U+FFFD,某处编码损坏')
  }

  // 7. 末尾没换行
  if (raw.length > 0 && !raw.endsWith('\n') && !raw.endsWith('\r')) {
    warnings.push('文件末尾没有换行符')
  }

  // 8. 单个段落特别长 (split 之后单段超过 25 句,大概就要看一眼)
  const cleaned = stripMarkdown(stripFrontmatter(raw))
  const rawParas = cleaned.split(/\n\s*\n+/).map((p) => p.trim()).filter(Boolean)
  const sentCounts = rawParas.map((p) => {
    const normalized = p.replace(/\s*\n\s*/g, ' ').trim()
    return splitSentencesEN(normalized).length
  })
  const tooLong = sentCounts
    .map((n, i) => ({ n, id: `p${i + 1}` }))
    .filter((x) => x.n > 25)
  if (tooLong.length > 0) {
    warnings.push(
      `split 后有 ${tooLong.length} 个超长段落 (>25 句): ${tooLong.map((x) => `${x.id}(${x.n})`).join(', ')}`,
    )
  }
  const tooShort = sentCounts
    .map((n, i) => ({ n, id: `p${i + 1}` }))
    .filter((x) => x.n === 0)
  if (tooShort.length > 0) {
    warnings.push(
      `split 后有 ${tooShort.length} 个段落 0 句 (可能是只剩标题/分隔线): ${tooShort.map((x) => x.id).join(', ')}`,
    )
  }

  // 输出
  console.log(`\n🔍 输入体检: ${path.relative(__dirname, filePath)}`)
  console.log(`   大小:      ${bytes} 字节 (${lines.length} 行, ${nonEmpty} 非空)`)
  if (issues.length === 0 && warnings.length === 0) {
    console.log(`   ✅ 没有发现异常`)
  } else {
    if (issues.length) {
      console.log(`   ❌ 严重 (${issues.length}):`)
      for (const i of issues) console.log(`      · ${i}`)
    }
    if (warnings.length) {
      console.log(`   ⚠️  提示 (${warnings.length}):`)
      for (const w of warnings) console.log(`      · ${w}`)
    }
  }
}

async function cmdSplit(): Promise<void> {
  if (!existsSync(INPUT_MD)) {
    throw new Error(`index.md 不存在: ${INPUT_MD}`)
  }
  const raw = await readFile(INPUT_MD, 'utf-8')
  validateInput(raw, INPUT_MD)
  const cleaned = stripMarkdown(stripFrontmatter(raw))

  // 段落 = 一个或多个空行分隔
  const rawParas = cleaned.split(/\n\s*\n+/).map((p) => p.trim()).filter(Boolean)

  const result: any = {
    metadata: {
      title: '',
      title_zh: '',
      author: '',
      author_zh: '',
      date: '',
      source: '',
      author_bio: '',
      preprocessing_notes: [
        '由 generate.ts split 步骤生成',
        '已去除 markdown 语法、保留文本',
      ],
    },
    intro: null,
    paragraphs: [] as any[],
    outro: null,
  }

  rawParas.forEach((p, pi) => {
    const id = `p${pi + 1}`
    // 段落内部换行折叠成空格
    const normalized = p.replace(/\s*\n\s*/g, ' ').trim()
    const sents = splitSentencesEN(normalized)
    result.paragraphs.push({
      id,
      sentences: sents.map((s, si) => ({ id: `${id}-s${si + 1}`, en: s })),
      summary: '',
    })
  })

  current = result
  await saveProject()

  // === 段落明细 ===
  const totalSent = result.paragraphs.reduce(
    (n: number, p: any) => n + p.sentences.length,
    0,
  )
  const sentCounts = result.paragraphs.map((p: any) => p.sentences.length)
  const maxSent = Math.max(...sentCounts)
  const minSent = Math.min(...sentCounts)
  const avgSent = totalSent / result.paragraphs.length

  console.log(`\n✅ Step 1: split 完成`)
  console.log(`   项目:       ${PROJECT}`)
  console.log(`   → ${path.relative(__dirname, OUTPUT_JSON)}`)
  console.log(`\n📊 段落统计:`)
  console.log(`   段落数:     ${result.paragraphs.length}`)
  console.log(`   句子总数:   ${totalSent}`)
  console.log(`   平均:       ${avgSent.toFixed(1)} 句/段`)
  console.log(`   最少:       ${minSent} 句`)
  console.log(`   最多:       ${maxSent} 句`)
  console.log(`\n📋 每段详情 (ID  句数  字符数):`)
  const idW = Math.max(4, ...result.paragraphs.map((p: any) => p.id.length))
  result.paragraphs.forEach((p: any, i: number) => {
    const chars = p.sentences.reduce((n: number, s: any) => n + s.en.length, 0)
    const flag = p.sentences.length === 0 ? ' ⚠️ 0 句' : ''
    console.log(
      `   ${p.id.padEnd(idW)}  ${String(p.sentences.length).padStart(3)} 句  ${String(chars).padStart(5)} 字${flag}`,
    )
  })
}

// ===== LLM 通用 =====

const LLM_ENDPOINT =
  process.env.MINIMAX_LLM_ENDPOINT || 'https://api.minimaxi.com/v1/chat/completions'
const LLM_MODEL = process.env.MINIMAX_LLM_MODEL || 'MiniMax-M3'

/**
 * --debug 时,把每次 LLM 调用的请求/响应追加到 LOG_FILE。
 * 任何写入失败都不会影响主流程。
 */
function logLLMCall(entry: {
  caller: string
  prompt: string
  rawBody: string
  extracted: string
  status: number | null
  durationMs: number
  error?: string
}): void {
  if (!(globalThis as any).__GEN_DEBUG__) return
  try {
    const sep = '='.repeat(80)
    const ts = new Date().toISOString()
    const parts = [
      sep,
      `[${ts}] ${entry.caller}`,
      `  endpoint: ${LLM_ENDPOINT}`,
      `  model:    ${LLM_MODEL}`,
      `  duration: ${entry.durationMs}ms`,
      `  status:   ${entry.status ?? '-'}`,
      entry.error ? `  ERROR:    ${entry.error}` : '',
      `  --- request ---`,
      entry.prompt,
      `  --- response (raw) ---`,
      entry.rawBody,
      `  --- response (extracted) ---`,
      entry.extracted,
      sep,
      '',
    ].filter((x) => x !== '')
    writeFileSync(LOG_FILE, parts.join('\n') + '\n', { flag: 'a' })
  } catch {
    // 写日志失败不影响主流程
  }
}

async function callLLM(prompt: string, caller = 'callLLM'): Promise<string> {
  const key = process.env.MINIMAX_API_KEY
  if (!key) throw new Error('MINIMAX_API_KEY 未设置,请检查 .env')

  const t0 = Date.now()
  let status: number | null = null
  let rawBody = ''
  let savedExtracted = ''

  try {
    const res = await fetch(LLM_ENDPOINT, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${key}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: LLM_MODEL,
      messages: [{ role: 'user', content: prompt }],
      max_completion_tokens: 4096,
      temperature: 0.7,
    }),
  })
  status = res.status
  if (!res.ok) {
    rawBody = await res.text().catch(() => '')
    throw new Error(`LLM ${res.status} ${res.statusText} · ${rawBody.slice(0, 300)}`)
  }
  const body = await res.json()
  rawBody = JSON.stringify(body)
  let content: string = body?.choices?.[0]?.message?.content ?? ''
  // 去 思考链 (MiniMax-M3 等模型会有)
  content = content.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
  // 去 ```json 等代码块包装
  content = content.replace(/^```(?:json)?\s*\n/i, '').replace(/\n```\s*$/, '').trim()
  savedExtracted = content
  return content
  } catch (e: any) {
    logLLMCall({
      caller,
      prompt,
      rawBody,
      extracted: '',
      status,
      durationMs: Date.now() - t0,
      error: e?.message ?? String(e),
    })
    throw e
  } finally {
    if (savedExtracted) {
      logLLMCall({
        caller,
        prompt,
        rawBody,
        extracted: savedExtracted,
        status,
        durationMs: Date.now() - t0,
      })
    }
  }
}

/** 容错抽取 JSON:在 LLM 自由文本里找到第一个 { 起、最后一个 } 止 */
function extractJSON(s: string): any {
  const first = s.indexOf('{')
  const last = s.lastIndexOf('}')
  if (first < 0 || last < 0) {
    throw new Error(
      `LLM 回复里找不到 JSON 对象: ${JSON.stringify(s.slice(0, 200))}`,
    )
  }
  const candidate = s.slice(first, last + 1)
  try {
    return JSON.parse(candidate)
  } catch (e: any) {
    // 把出错位置往前 60 字 / 往后 60 字打出来,方便定位
    const m = e.message?.match(/position (\d+)/i)
    const pos = m ? Number(m[1]) : -1
    const lo = Math.max(0, pos - 60)
    const hi = Math.min(candidate.length, pos + 60)
    const snippet =
      pos >= 0
        ? `${candidate.slice(0, lo)}⛔${candidate.slice(lo, hi)}⛔${candidate.slice(hi)}`
        : candidate.slice(0, 200)
    throw new Error(
      `JSON.parse 失败: ${e.message}\n出错片段: ${snippet}\n原始回复(前 600 字): ${JSON.stringify(s.slice(0, 600))}`,
    )
  }
}

// ===== Step 2: intro / outro =====

async function cmdIntro(): Promise<void> {
  const data = await loadProject()
  if (!data.paragraphs?.length) throw new Error('paragraphs 为空,请先 split')

  const fullText = data.paragraphs
    .map((p: any) => p.sentences.map((s: any) => s.en).join(' '))
    .join('\n\n')

  const prompt = `你是一档英文精读视频节目的撰稿人,负责把英文文章做成给中国读者的英文学习视频。下面是文章全文:

<article>
${fullText.slice(0, 8000)}
</article>

请严格输出 JSON,不要 \`\`\`json 代码块包装,不要任何解释性文字,只输出 JSON 本身。

硬性规则:
1. 字符串里**禁止出现未转义的英文双引号 "**;如需在中文里引用词句,统一用中文「」或反引号 \`code\` 替代
2. 字符串里如果真的需要英文双引号,必须写成 \\\"
3. 字符串内部允许普通换行 (\\n)

{
  "intro": "<中文 3-5 段:文章主题、作者背景、写作动机、读者将学到什么、有钩子有趣味>",
  "outro": "<中文 3-5 段:全文要点总结升华、给读者的行动建议>"
}
`
  const content = await callLLM(prompt)
  const parsed = extractJSON(content)
  data.intro = parsed.intro ?? data.intro
  data.outro = parsed.outro ?? data.outro

  await saveProject()

  console.log(`✅ Step 2: intro/outro 完成`)
  console.log(`   intro: ${data.intro?.length ?? 0} 字`)
  console.log(`   outro: ${data.outro?.length ?? 0} 字`)
}

// ===== Step 3: sentence / summary =====

async function generateSentence(
  s: any,
): Promise<{ translation: string; explanation: string }> {
  const prompt = `你给一段英文做精读讲解。

英文:
${s.en}

请严格输出 JSON (不要 \`\`\`json 包装、不要解释文字):
{
  "translation": "<中文翻译,自然流畅、尊重原文、结合上下文>",
  "explanation": "<讲解正文 100-300 字:重点单词短语(用引号标注,例如 \"phrasal verbs\")、语法点(如虚拟语气/倒装/从句嵌套)、背景知识、文化梗。生动有趣,假设读者英语不太好>"
}
`
  return extractJSON(await callLLM(prompt))
}

async function generateSummary(p: any): Promise<string> {
  const list = p.sentences.map((s: any, i: number) => `${i + 1}. ${s.en}`).join('\n')
  const prompt = `你给一段英文文章的一个段落写一句段尾总结 (80-150 字),要点出该段在全文中的位置和关键论点,避免空泛。

段落句子:
${list}

请严格输出 JSON:
{
  "summary": "..."
}
`
  const { summary } = extractJSON(await callLLM(prompt))
  return summary ?? ''
}

/**
 * 处理一个段落的所有句子 + summary。
 * - 已填过 translation+explanation 的跳过 (idempotent,断点续跑)
 * - 每句成功都立刻落盘
 */
async function processParagraph(p: any): Promise<void> {
  if (!p.sentences?.length) return

  for (let i = 0; i < p.sentences.length; i++) {
    const s = p.sentences[i]
    if (s.translation && s.explanation) {
      process.stdout.write('·')
      continue
    }
    process.stdout.write(s.id + ' ')
    try {
      const r = await generateSentence(s)
      s.translation = r.translation
      s.explanation = r.explanation
      await saveProject()
    } catch (e: any) {
      console.error(`\n  [WARN] ${s.id} 失败: ${e.message}`)
    }
  }
  if (!p.summary) {
    process.stdout.write(` ${p.id}/summary`)
    try {
      p.summary = await generateSummary(p)
      await saveProject()
    } catch (e: any) {
      console.error(`\n  [WARN] ${p.id} summary 失败: ${e.message}`)
    }
  }
  process.stdout.write('\n')
}

async function cmdSentencesAll(): Promise<void> {
  const data = await loadProject()
  console.log(`📝 Step 3: 全部段落 (${data.paragraphs.length} 段)`)
  for (const p of data.paragraphs) {
    await processParagraph(p)
  }
  console.log(`✅ Step 3: 全部完成`)
}

async function cmdSentenceOne(target: string): Promise<void> {
  if (!target) throw new Error('单段模式请指定段落 id,如 p9')
  const data = await loadProject()
  const p = data.paragraphs.find((x: any) => x.id === target)
  if (!p) {
    const list = data.paragraphs.map((x: any) => x.id).join(', ')
    throw new Error(`找不到段落 ${target};现有段落: ${list}`)
  }
  console.log(`📝 Step 3: 单段 ${target}`)
  await processParagraph(p)
  console.log(`✅ Step 3: 段落 ${target} 完成`)
}

// ===== Main =====

async function main(): Promise<void> {
  switch (CMD) {
    case 'split':
      return cmdSplit()
    case 'intro':
      return cmdIntro()
    case 'sentences':
      return cmdSentencesAll()
    case 'sentence':
      return cmdSentenceOne(TARGET || '')
    default:
      console.error(`❌ 未知命令: ${CMD}`)
      printUsage()
      process.exit(1)
  }
}

main().catch((e) => {
  console.error('❌', e?.stack || e?.message || e)
  process.exit(1)
})
