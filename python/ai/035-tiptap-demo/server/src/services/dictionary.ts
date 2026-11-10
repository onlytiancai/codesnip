/**
 * 查词服务：本地词库查一个英文单词。
 *
 * 词库是 **JSONL**（一行一个 JSON 对象），来自 024-cet4 项目的 vocab_all.json：
 *   {wordRank, headWord, content: {word: {wordHead, wordId, content: {...}}}, bookId}
 * 里面真正要的东西都在 content.word.content 里：
 *   trans[]  中文释义、phrase 短语、sentence 例句、syno 同近义词、antos 反义词、remMethod 记忆
 *
 * **为什么用「字节偏移索引」而不是全量载入**：
 * 那个文件 12MB / 5011 行。全量 parse 进内存是一次性 ~12MB 的常驻开销；
 * 这里只在建索引时扫一遍（每行 parse 一次、只取 wordHead、马上丢掉），
 * 之后查词 = seek 到那一行 → readline → parse 这一行（~2.5KB）。
 * 内存里只留 5011 条 Map<string, number>。
 * 思路和 024-cet4 的 scripts/cet4.py 一致。
 */

import { open, readFile, stat } from 'node:fs/promises'
import { envValue, loadEnv } from '../config/env.js'

/** 没配 VOCAB_PATH 时的兜底路径。 */
export const DEFAULT_VOCAB_PATH =
  '/Users/huhao/src/codesnip/python/ai/024-cet4/vocab_all.json'

/** 词条 —— 从原始记录里裁出来的、给前端看的形状。 */
export interface DictionaryEntry {
  word: string
  /** 美音 / 英音音标（不带斜杠） */
  usphone?: string
  ukphone?: string
  /** 词性释义 */
  trans: { pos: string; cn: string; en: string }[]
  /** 例句 */
  sentences: { en: string; cn: string }[]
  /** 短语 */
  phrases: { en: string; cn: string }[]
  /** 同近义词 */
  synonyms: { pos: string; tran: string; words: string[] }[]
  /** 反义词 */
  antonyms: string[]
  /** 记忆法 */
  mnemonic?: string
}

export interface DictionaryStatus {
  /** 当前用的词库路径 */
  path: string
  /** 索引里有多少个词 */
  count: number
  /** 索引建好了吗 */
  loaded: boolean
  /** 建索引时出的问题（文件不存在 / 解析失败） */
  error?: string
}

export interface DictionaryService {
  lookup(word: string): Promise<DictionaryEntry | undefined>
  /** 查不到时给的相近词（词库里的词头），点一下就能看那个词的解释 */
  suggest(word: string, limit?: number): Promise<string[]>
  status(): DictionaryStatus
  /** 重新读 .env + 重建索引 */
  reload(): Promise<DictionaryStatus>
}

export interface CreateDictionaryServiceOptions {
  /** 覆盖词库路径（测试用）；不传就读 process.env.VOCAB_PATH */
  path?: string
}

/** 词库路径：env 优先，代码默认值兜底。 */
export function vocabPath(explicit?: string): string {
  return explicit || envValue('VOCAB_PATH') || DEFAULT_VOCAB_PATH
}

// ---------------------------------------------------------------------------
// 原始记录 → DictionaryEntry
// ---------------------------------------------------------------------------

interface RawTrans {
  tranCn?: unknown
  tranOther?: unknown
  pos?: unknown
}

interface RawSentence {
  sContent?: unknown
  sCn?: unknown
}

interface RawPhrase {
  pContent?: unknown
  pCn?: unknown
}

interface RawSyno {
  pos?: unknown
  tran?: unknown
  hwds?: { w?: unknown }[]
}

interface RawRecord {
  content?: {
    word?: {
      wordHead?: unknown
      content?: {
        usphone?: unknown
        ukphone?: unknown
        trans?: RawTrans[]
        sentence?: { sentences?: RawSentence[] }
        phrase?: { phrases?: RawPhrase[] }
        syno?: { synos?: RawSyno[] }
        /** 反义词 —— 注意是 `antos`（词库里 236 个词条都这么写），不是 `anto` */
        antos?: { anto?: { hwd?: unknown }[] }
        remMethod?: { val?: unknown }
      }
    }
  }
}

function str(v: unknown): string | undefined {
  return typeof v === 'string' && v.trim() ? v.trim() : undefined
}

function arr(v: unknown): unknown[] {
  return Array.isArray(v) ? v : []
}

/**
 * 词库只收词头（abandon），不收变形（abandons）。
 * 查不到的时候退一步：试试常见的变形还原，命中就返回，
 * 同时告诉前端「你查的是 abandons，词库里是 abandon」。
 *
 * 只认规则后缀 —— children → child 这种不规则复数还原不出来，
 * 那类词靠 suggest() 给的相近词兜底。
 */
export function stemVariants(word: string): string[] {
  const w = word.toLowerCase()
  const out: string[] = []
  const push = (s: string) => {
    if (s.length >= 3 && s !== w && !out.includes(s)) out.push(s)
  }
  if (w.endsWith('ies')) push(w.slice(0, -3) + 'y')
  if (w.endsWith('es')) push(w.slice(0, -2))
  if (w.endsWith('s')) push(w.slice(0, -1))
  if (w.endsWith('ed')) {
    push(w.slice(0, -2))
    push(w.slice(0, -1))
  }
  if (w.endsWith('ing')) {
    push(w.slice(0, -3))
    push(w.slice(0, -3) + 'e')
  }
  return out
}

/** 建议最多给几个词。 */
export const DEFAULT_SUGGEST_LIMIT = 8

/** 至少要多长的公共前缀才算「相近」。 */
const MIN_PREFIX = 3

/** 「公共前缀相关」那一档最多给几个。 */
const PREFIX_LIMIT = 5

/** 「编辑距离最短」那一档给几个。 */
const DISTANCE_LIMIT = 5

/** 两个词的公共前缀长度。 */
function commonPrefixLength(a: string, b: string): number {
  const max = Math.min(a.length, b.length)
  let i = 0
  while (i < max && a[i] === b[i]) i++
  return i
}

/**
 * 编辑距离（OSA / Damerau-Levenshtein）—— 判断拼写有多接近。
 *
 * 比普通 Levenshtein 多算一种编辑：**相邻字符换位**算 1 次。
 * "recieve" → "receive" 就是 ie / ei 换位，按普通距离算是 2，
 * 换成 1 才能让它排到 "recipe" 前面去。
 */
function editDistance(a: string, b: string): number {
  if (a === b) return 0
  if (!a.length) return b.length
  if (!b.length) return a.length

  const d: number[][] = []
  for (let i = 0; i <= a.length; i++) d.push([i, ...new Array<number>(b.length).fill(0)])
  for (let j = 0; j <= b.length; j++) d[0][j] = j

  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
      // 相邻两个字符换位
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1)
      }
    }
  }
  return d[a.length][b.length]
}

/**
 * 在词库的词头里找和 word 相近的词。
 *
 * 给两档，合起来去重：
 *
 *  1. **公共前缀相关**（前缀至少 3 个字符）—— 真正的「相近的词」。
 *     "scenarios" → scene / scenery。只按编辑距离排的话，
 *     "scenarios" 会被匹配到 "serious" / "sunrise" 这种毫不相干的词。
 *     这一档内部的排序看：词头是不是查询词的前缀（查 scenes → scene）、
 *     公共前缀多长、拼写有多接近。
 *  2. **编辑距离最短的 5 个** —— 拼错的词靠它兜底。
 *     "recieve" → receive（相邻字符换位，OSA 距离算 1）。
 *
 * 两档都拿**用户实际打的那个词**去比（不拿词干比）：
 * 词干短，比出来的距离会失真 —— 拿 "scenario" 去比，
 * "scar" 看着只差 1，可用户明明打的是 "scenarios"。
 */
export function suggestSimilar(
  heads: Iterable<string>,
  word: string,
  limit = DEFAULT_SUGGEST_LIMIT,
): string[] {
  const key = word.trim().toLowerCase()
  if (!key) return []

  interface Scored {
    head: string
    /** 公共前缀长度 */
    cp: number
    /** 编辑距离 */
    dist: number
    /** 综合分，越高越像 */
    score: number
  }

  const scored: Scored[] = []
  for (const head of heads) {
    if (head === key) continue // 别把自己推给自己
    const cp = commonPrefixLength(head, key)
    const dist = editDistance(head, key)
    let score = cp * 20 - dist * 40 - Math.abs(head.length - key.length) * 5
    if (key.startsWith(head)) score += 1000 // 词头是查询词的前缀
    else if (head.startsWith(key)) score += 800 // 词头是查询词的延伸
    scored.push({ head, cp, dist, score })
  }

  // 第一档：公共前缀相关的
  const byPrefix = scored
    .filter((s) => s.cp >= MIN_PREFIX)
    .sort((a, b) => b.score - a.score || a.head.localeCompare(b.head))
    .slice(0, PREFIX_LIMIT)

  // 第二档：编辑距离最短的 5 个（距离相同时按综合分排）
  const byDistance = [...scored]
    .sort((a, b) => a.dist - b.dist || b.score - a.score || a.head.localeCompare(b.head))
    .slice(0, DISTANCE_LIMIT)

  // 合并去重：第一档在前，第二档补后面
  const out: string[] = []
  for (const s of [...byPrefix, ...byDistance]) {
    if (!out.includes(s.head)) out.push(s.head)
  }
  return out.slice(0, limit)
}

/** 从一条原始记录里抽出我们要的字段。 */
export function toEntry(raw: unknown): DictionaryEntry | undefined {
  const rec = raw as RawRecord
  const word = rec?.content?.word
  if (!word) return undefined
  const head = str(word.wordHead)
  if (!head) return undefined
  const c = word.content ?? {}

  return {
    word: head,
    usphone: str(c.usphone),
    ukphone: str(c.ukphone),
    trans: arr(c.trans).map((t) => ({
      pos: str((t as RawTrans).pos) ?? '',
      cn: str((t as RawTrans).tranCn) ?? '',
      en: str((t as RawTrans).tranOther) ?? '',
    })),
    sentences: arr(c.sentence?.sentences).map((s) => ({
      en: str((s as RawSentence).sContent) ?? '',
      cn: str((s as RawSentence).sCn) ?? '',
    })),
    phrases: arr(c.phrase?.phrases).map((p) => ({
      en: str((p as RawPhrase).pContent) ?? '',
      cn: str((p as RawPhrase).pCn) ?? '',
    })),
    synonyms: arr(c.syno?.synos).map((s) => ({
      pos: str((s as RawSyno).pos) ?? '',
      tran: str((s as RawSyno).tran) ?? '',
      words: arr((s as RawSyno).hwds)
        .map((h) => str((h as { w?: unknown }).w))
        .filter((w): w is string => Boolean(w)),
    })),
    antonyms: arr(c.antos?.anto)
      .map((a) => str((a as { hwd?: unknown }).hwd))
      .filter((w): w is string => Boolean(w)),
    mnemonic: str(c.remMethod?.val),
  }
}

// ---------------------------------------------------------------------------
// 服务
// ---------------------------------------------------------------------------

/**
 * 建索引：wordHead（小写）→ 该行在文件里的字节偏移。
 *
 * 按字节扫（不是按行读文本）—— 这样偏移量是精确的，不受 CRLF / 多字节字符影响。
 * 整个文件读成一个 Buffer（12MB，一次性的），扫完就释放；
 * 每行只 parse 一次取 wordHead，然后立刻丢掉解析结果 ——
 * 12MB 的原始记录不会常驻内存。
 */
export async function buildIndex(
  path: string,
): Promise<Map<string, number>> {
  const index = new Map<string, number>()
  const buf = await readFile(path)

  let start = 0
  for (;;) {
    const nl = buf.indexOf(0x0a, start)
    const end = nl === -1 ? buf.length : nl
    if (end > start) {
      const line = buf.toString('utf8', start, end)
      if (line.trim()) {
        try {
          const parsed = JSON.parse(line) as RawRecord
          const head = str(parsed?.content?.word?.wordHead)
          // 重复词头保留第一次出现的（和 024-cet4 的 build_index 一致）
          if (head) {
            const key = head.toLowerCase()
            if (!index.has(key)) index.set(key, start)
          }
        } catch {
          // 坏行跳过 —— 索引少一个词，总比整个词库不可用强
        }
      }
    }
    if (nl === -1) break
    start = nl + 1
  }
  return index
}

/**
 * 按偏移读回那一行并解析。
 *
 * 一行可能好几 KB，所以循环读，直到找到换行符为止。
 */
export async function readRow(
  path: string,
  offset: number,
): Promise<unknown | undefined> {
  const handle = await open(path, 'r')
  try {
    const chunks: Buffer[] = []
    let position = offset
    let line: string | undefined
    for (;;) {
      const buf = Buffer.alloc(32 * 1024)
      const { bytesRead } = await handle.read(buf, 0, buf.length, position)
      if (bytesRead === 0) {
        const text = Buffer.concat(chunks).toString('utf8')
        line = text
        break
      }
      const slice = buf.subarray(0, bytesRead)
      const nl = slice.indexOf(0x0a)
      if (nl !== -1) {
        chunks.push(slice.subarray(0, nl))
        line = Buffer.concat(chunks).toString('utf8')
        break
      }
      chunks.push(slice)
      position += bytesRead
    }
    if (!line || !line.trim()) return undefined
    return JSON.parse(line)
  } finally {
    await handle.close()
  }
}

export function createDictionaryService(
  options: CreateDictionaryServiceOptions = {},
): DictionaryService {
  let path = vocabPath(options.path)
  let index = new Map<string, number>()
  let loaded = false
  let error: string | undefined
  /** 正在进行的建索引任务 —— lookup 等它，reload 会换掉它 */
  let building: Promise<void> | null = null

  async function build(): Promise<void> {
    building = (async () => {
      try {
        await stat(path)
      } catch {
        index = new Map()
        loaded = false
        error = `词库文件不存在：${path}`
        return
      }
      try {
        index = await buildIndex(path)
        loaded = true
        error = undefined
      } catch (err) {
        index = new Map()
        loaded = false
        error = `词库读取失败：${err instanceof Error ? err.message : String(err)}`
      }
    })()
    await building
  }

  // 服务一创建就开始建索引（不 await）—— server 启动不被 12MB 扫描拖住
  void build()

  function status(): DictionaryStatus {
    return { path, count: index.size, loaded, error }
  }

  return {
    async lookup(word) {
      const key = word.trim().toLowerCase()
      if (!key) return undefined
      await building
      if (!loaded) return undefined

      const read = async (k: string): Promise<DictionaryEntry | undefined> => {
        const offset = index.get(k)
        if (offset === undefined) return undefined
        try {
          const raw = await readRow(path, offset)
          return toEntry(raw)
        } catch {
          return undefined
        }
      }

      const exact = await read(key)
      if (exact) return exact

      // 词库只收词头 —— 变形（children / studied）退一步试试还原
      for (const variant of stemVariants(key)) {
        const hit = await read(variant)
        if (hit) return hit
      }
      return undefined
    },

    async suggest(word, limit) {
      await building
      if (!loaded) return []
      return suggestSimilar(index.keys(), word, limit)
    },

    status,

    async reload() {
      // 重新读 .env —— 改了 VOCAB_PATH 不用重启 server
      try {
        loadEnv()
      } catch (err) {
        error = `.env 解析失败：${err instanceof Error ? err.message : String(err)}`
      }
      path = vocabPath(options.path)
      await build()
      return status()
    },
  }
}
