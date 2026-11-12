import type { Node as PMNode } from '@tiptap/pm/model'
import type { EditorState } from '@tiptap/pm/state'

/**
 * 写作上下文：段落 / 句子 / 单词 的定位，以及「文本偏移 ↔ ProseMirror 位置」的换算。
 *
 * 这里算的东西有两个出口：
 *  1. 编辑器每次事务后上报「当前句子的全局索引」→ EditorPage 用它做双栏对齐
 *  2. highlightContext.ts 的装饰扩展用它算 Decoration 的位置
 *
 * 两边共用同一套算法，所以「高亮」和「索引」永远不会各算各的。
 *
 * ⚠️ 偏移口径：`splitFn` 切的是**段落纯文本**（= node.textContent），
 * 位置口径是 **doc 坐标**。两者不是一回事 —— hardBreak（Shift+Enter）在 textContent 里
 * 不占字符，但在 doc 里占 1 个位置。所以不能直接 `段落起点 + 偏移`，
 * 得先经过下面的 Chunk 映射。
 */

export interface Sentence {
  text: string
  start: number
  end: number
}

/**
 * 常见缩写 —— 这些后面的点不是句号。
 * （比较时统一小写）
 */
const ABBREVIATIONS = new Set([
  'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'rev', 'vs', 'etc',
  'al', 'inc', 'ltd', 'co', 'corp', 'dept', 'est', 'fig', 'no', 'vol',
  'approx', 'univ', 'assn', 'e.g', 'i.e', 'a.m', 'p.m', 'u.s', 'u.k',
])

/** 句号后面可能跟的右括号 / 右引号 —— 判断「下一个字符」时先跳过它们。 */
const CLOSING = new Set([')', ']', '}', '»', '”', '’', '"', "'"])

/** 中日韩字符（含全角标点）—— 中文不用空格，所以「他好.然后」也算断句。 */
const CJK = /[⺀-鿿豈-﫿＀-￯]/

/** 英文的句末标点（换行也算边界）。 */
const ENGLISH_BOUNDARIES = new Set(['.', '!', '?', '\n'])

/** 中文的句末标点 —— 中文里也常有半角的 . ! ?，一并认。 */
const CHINESE_BOUNDARIES = new Set(['。', '！', '？', '.', '!', '?', '\n'])

/**
 * text[i] 这个点是不是句末的句号。
 *
 * 两条规则：
 *
 *  1. **点后面（跳过右括号右引号）得是空白、中日韩字符、或者文本结尾** ——
 *     "PHP8.4"、"3.14"、"www.example.com" 里的点都不是断句。
 *     这一条把绝大多数「点被误当句号」的情况挡掉了。
 *  2. **点前面那个词不能是缩写** ——
 *     "Mr. Smith"、"e.g. this"、"the U.S. economy" 里的点不是句号。
 */
function isEndDot(text: string, i: number): boolean {
  // 1. 后面必须是空白、中日韩字符或结尾
  let j = i + 1
  while (j < text.length && CLOSING.has(text[j])) j++
  if (j < text.length && !/\s/.test(text[j]) && !CJK.test(text[j])) return false

  // 2. 前面那个词不能是缩写
  const m = /([A-Za-z][A-Za-z.]*)$/.exec(text.slice(0, i))
  if (m) {
    const token = m[1].toLowerCase()
    // 带点的缩写（U.S / e.g）或者常见缩写（Mr / etc）
    if (token.includes('.') || ABBREVIATIONS.has(token)) return false
  }
  return true
}

/**
 * 按句末标点切句。
 *
 * 换行是边界但**不算进句子**（和以前的正则行为一致）；
 * 其余标点算进前一句。
 */
function splitSentences(text: string, boundaries: Set<string>): Sentence[] {
  const out: Sentence[] = []
  let start = 0

  const push = (from: number, to: number) => {
    const s = text.slice(from, to)
    // 整段都是标点（开头连着的标点）就跳过
    let hasContent = false
    for (const ch of s) {
      if (!boundaries.has(ch)) {
        hasContent = true
        break
      }
    }
    if (hasContent) out.push({ text: s, start: from, end: to })
  }

  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (!boundaries.has(ch)) continue
    if (ch === '.' && !isEndDot(text, i)) continue
    // 换行不进句子，其余标点归前一句
    push(start, ch === '\n' ? i : i + 1)
    start = i + 1
  }
  if (start < text.length) push(start, text.length)

  return out
}

/**
 * 英文切句（. ! ?）。hardBreak / 换行也视为边界。
 *
 * 点不是见到就断句 —— "PHP8.4"、"Mr. Smith"、"www.a.com" 都不会被劈开。
 */
export function splitEnglishSentences(text: string): Sentence[] {
  return splitSentences(text, ENGLISH_BOUNDARIES)
}

/**
 * 中文切句（。！？.!?）。同样把 hardBreak / 换行视为边界。
 *
 * 中文里也常有 "PHP8.4"、"3.14" 这种带点的串，点同样不乱断。
 */
export function splitChineseSentences(text: string): Sentence[] {
  return splitSentences(text, CHINESE_BOUNDARIES)
}

export type SplitFn = (text: string) => Sentence[]

/**
 * 在 text 中定位包含 offset 的"句子"。返回句子的 text/start/end + 索引。
 */
export function findSentenceInfo(
  offset: number,
  sentences: Sentence[],
): { info: Sentence; idx: number } | null {
  for (let i = 0; i < sentences.length; i++) {
    const s = sentences[i]
    if (offset >= s.start && offset <= s.end) {
      return { info: s, idx: i }
    }
    if (s.start > offset) break
  }
  return null
}

/**
 * 在 text 中定位包含 offset 的"单词"。
 * 单词 = 连续的非空白字符。
 */
export function findWordAt(
  text: string,
  offset: number,
): { text: string; start: number; end: number } | null {
  const re = /\S+/g
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    const start = m.index
    const end = start + m[0].length
    if (offset >= start && offset <= end) {
      return { text: m[0], start, end }
    }
    if (start > offset) break
  }
  return null
}

// ===== 位置几何 =====

/**
 * 段落里的一个片段。
 * - text：这段的纯文本（叶子节点如 hardBreak 不贡献字符）
 * - from/to：这段在 doc 里的位置区间 [from, to)
 */
export interface Chunk {
  text: string
  from: number
  to: number
}

/**
 * 把段落摊平成「文本片段 + doc 位置」。
 *
 * 纯文本节点的 text 拼起来 === para.textContent（切句口径不变），
 * 但 doc 位置是对的 —— 叶子节点（hardBreak）虽然不含字符，仍占 1 个位置。
 */
export function buildChunks(para: PMNode, paraStart: number): Chunk[] {
  const chunks: Chunk[] = []
  para.forEach((child, offset) => {
    const from = paraStart + 1 + offset
    chunks.push({
      text: child.isText ? (child.text ?? '') : '',
      from,
      to: from + child.nodeSize,
    })
  })
  return chunks
}

/** 片段纯文本拼起来（=== node.textContent） */
export function chunkText(chunks: Chunk[]): string {
  let text = ''
  for (const c of chunks) text += c.text
  return text
}

/**
 * 找出「文本偏移」落在哪个片段上。
 * 返回 -1 表示越界。
 *
 * prefer='end'   → 取**第一个**命中的片段（边界落在前一段末尾）
 * prefer='start' → 取**最后一个**命中的片段（边界落在后一段开头）
 *
 * 两者只在片段边界上不同 —— 那正是 hardBreak 造成的位置空隙所在，
 * 区间映射用 from='start' / to='end' 就能自动跨过这个空隙。
 */
function findChunkIndex(chunks: Chunk[], offset: number, prefer: 'start' | 'end'): number {
  let acc = 0
  let found = -1
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i]
    if (!c.text.length) continue
    const start = acc
    const end = acc + c.text.length
    if (offset >= start && offset <= end) {
      found = i
      if (prefer === 'end') break
    }
    acc = end
  }
  return found
}

/**
 * 段落内的「文本偏移」→ doc 位置。
 */
export function posAt(chunks: Chunk[], offset: number, prefer: 'start' | 'end'): number {
  const i = findChunkIndex(chunks, offset, prefer)
  if (i < 0) {
    // 越界（正常不会发生）：落到最后一个有文本的片段末尾
    let last = -1
    for (let k = 0; k < chunks.length; k++) if (chunks[k].text.length) last = k
    return last >= 0 ? chunks[last].to : 0
  }
  let acc = 0
  for (let k = 0; k < i; k++) acc += chunks[k].text.length
  return chunks[i].from + (offset - acc)
}

/**
 * doc 位置 → 段落内的「文本偏移」（光标位置换算用）。
 */
export function textOffsetAt(chunks: Chunk[], pos: number): number {
  let acc = 0
  for (const c of chunks) {
    if (!c.text.length) continue
    if (pos >= c.from && pos <= c.to) return acc + (pos - c.from)
    acc += c.text.length
  }
  return acc
}

/**
 * doc 里所有顶层块节点的内容区间（不含节点自身的开闭位置）。
 *
 * 给 `insertContentAt` 替换段落内容用 ——
 * from = 节点起点 +1，to = 节点终点 -1，正好是段落里的内容。
 */
export function topBlockContentRanges(doc: PMNode): { from: number; to: number }[] {
  const ranges: { from: number; to: number }[] = []
  doc.forEach((node, offset) => {
    ranges.push({ from: offset + 1, to: offset + node.nodeSize - 1 })
  })
  return ranges
}

// ===== 句子收集 =====

/**
 * 一句话在 doc 里的位置。
 */
export interface SentenceSpan {
  /** 文档级全局句子索引（0-based，跨段落累计） */
  index: number
  /** 所在段落节点的区间（给 Decoration.node 用） */
  paraFrom: number
  paraTo: number
  /** 这句话的区间（给 Decoration.inline 用） */
  from: number
  to: number
  text: string
}

/**
 * 把 doc 里的所有句子按顺序收集起来，附全局索引。
 *
 * 只数**顶层**的 paragraph / heading —— 别的块类型（列表、代码块…）
 * 不参与编号，这样「编辑器里的句子序号」和「跨栏对齐用的序号」是同一套。
 */
export function collectSentences(doc: PMNode, splitFn: SplitFn): SentenceSpan[] {
  const out: SentenceSpan[] = []
  let index = 0
  doc.forEach((node, offset) => {
    const name = node.type.name
    if (name !== 'paragraph' && name !== 'heading') return
    const paraFrom = offset
    const paraTo = offset + node.nodeSize
    const chunks = buildChunks(node, paraFrom)
    for (const s of splitFn(chunkText(chunks))) {
      out.push({
        index: index++,
        paraFrom,
        paraTo,
        from: posAt(chunks, s.start, 'start'),
        to: posAt(chunks, s.end, 'end'),
        text: s.text,
      })
    }
  })
  return out
}

// ===== 光标上下文 =====

/**
 * 光标所在位置的三层上下文（段落 / 句子 / 单词），全部是 doc 位置。
 */
export interface ContextSpan {
  /** 段落节点区间（Decoration.node） */
  paraFrom: number
  paraTo: number
  /** 句子区间（Decoration.inline） */
  sentFrom: number
  sentTo: number
  /** 单词区间（Decoration.inline） */
  wordFrom: number
  wordTo: number
  paragraph: string
  sentence: string
  word: string
  /** 文档级全局句子索引；光标不在任何句子里时是 -1 */
  sentenceIndex: number
}

/**
 * 从光标位置反推当前段落 / 句子 / 单词。
 *
 * 段落 = 光标所在的 paragraph 或 heading 节点。
 * 句子 = 段落里按标点切出来的、包含光标的那句。
 * 单词 = 句子里包含光标的那段连续非空白字符。
 */
export function resolveContext(state: EditorState, splitFn: SplitFn): ContextSpan | null {
  const { $from } = state.selection
  const doc = state.doc

  // 1. 找到最近的段落级节点
  let depth = -1
  for (let d = $from.depth; d > 0; d--) {
    const name = $from.node(d).type.name
    if (name === 'paragraph' || name === 'heading') {
      depth = d
      break
    }
  }
  if (depth === -1) return null

  const para = $from.node(depth)
  const paraFrom = $from.before(depth)
  const paraTo = paraFrom + para.nodeSize
  const chunks = buildChunks(para, paraFrom)
  const paragraph = chunkText(chunks)

  // 2. 光标在段落纯文本里的偏移
  const offset = textOffsetAt(chunks, $from.pos)

  // 3. 在段落中找包含光标的句子
  const sentences = splitFn(paragraph)
  const found = findSentenceInfo(offset, sentences)
  if (!found) {
    // 光标不在任何句子里（空段落）→ 没有句子索引
    return {
      paraFrom,
      paraTo,
      sentFrom: $from.pos,
      sentTo: $from.pos,
      wordFrom: $from.pos,
      wordTo: $from.pos,
      paragraph,
      sentence: '',
      word: '',
      sentenceIndex: -1,
    }
  }

  // 4. 在句子中找包含光标的单词
  const wordInfo = findWordAt(found.info.text, offset - found.info.start)

  const sentFrom = posAt(chunks, found.info.start, 'start')
  const sentTo = posAt(chunks, found.info.end, 'end')

  return {
    paraFrom,
    paraTo,
    sentFrom,
    sentTo,
    wordFrom: wordInfo ? posAt(chunks, found.info.start + wordInfo.start, 'start') : sentFrom,
    wordTo: wordInfo ? posAt(chunks, found.info.start + wordInfo.end, 'end') : sentFrom,
    paragraph,
    sentence: found.info.text,
    word: wordInfo?.text ?? '',
    sentenceIndex: countSentencesBefore(doc, paraFrom, splitFn) + found.idx,
  }
}

/**
 * 本段落之前所有顶层 paragraph / heading 里的句子总数。
 */
function countSentencesBefore(doc: PMNode, paraFrom: number, splitFn: SplitFn): number {
  let count = 0
  doc.forEach((node, offset) => {
    if (offset >= paraFrom) return
    const name = node.type.name
    if (name !== 'paragraph' && name !== 'heading') return
    count += splitFn(node.textContent).length
  })
  return count
}

/**
 * 光标所在句子的全局索引（跨面板对齐的唯一依据）。
 * 光标不在任何段落/句子里时是 -1。
 */
export function getSentenceIndex(state: EditorState, splitFn: SplitFn): number {
  return resolveContext(state, splitFn)?.sentenceIndex ?? -1
}
