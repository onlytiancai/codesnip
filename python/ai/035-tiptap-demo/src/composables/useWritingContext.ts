import type { Editor } from '@tiptap/core'

/**
 * 当前光标所在的写作上下文。
 * - paragraph: 整个段落文本
 * - paragraphHlStart/End: 当前句子在段落中的起止偏移（用于高亮）
 * - sentence: 当前句子文本
 * - sentenceHlStart/End: 当前单词在句子中的起止偏移
 * - word: 当前光标所在的单词（去尾随标点前的部分）
 * - sentenceIndex: 当前句子在文档中的全局索引（0-based，跨段落累计）
 */
export interface WritingContext {
  paragraph: string
  paragraphHlStart: number
  paragraphHlEnd: number
  sentence: string
  sentenceHlStart: number
  sentenceHlEnd: number
  word: string
  sentenceIndex: number
}

export const EMPTY_CONTEXT: WritingContext = {
  paragraph: '',
  paragraphHlStart: 0,
  paragraphHlEnd: 0,
  sentence: '',
  sentenceHlStart: 0,
  sentenceHlEnd: 0,
  word: '',
  sentenceIndex: -1,
}

export interface Sentence {
  text: string
  start: number
  end: number
}

/**
 * 英文切句（. ! ?）。hardBreak / 换行也视为边界。
 */
export function splitEnglishSentences(text: string): Sentence[] {
  return splitByRegex(text, /[^.!?\n]+[.!?]?/g)
}

/**
 * 中文切句（。！？.!?）。同样把 hardBreak / 换行视为边界。
 */
export function splitChineseSentences(text: string): Sentence[] {
  return splitByRegex(text, /[^。！？.!?\n]+[。！？.!?]?/g)
}

function splitByRegex(text: string, regex: RegExp): Sentence[] {
  const out: Sentence[] = []
  regex.lastIndex = 0
  let m: RegExpExecArray | null
  while ((m = regex.exec(text)) !== null) {
    const start = m.index
    const end = start + m[0].length
    out.push({ text: m[0], start, end })
    if (m[0].length === 0) regex.lastIndex++ // 防 zero-width 死循环
  }
  return out
}

/**
 * 在 text 中定位包含 offset 的"句子"。返回句子的 text/start/end + 索引。
 */
function findSentenceInfo(
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
function findWordAt(
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

export type SplitFn = (text: string) => Sentence[]

/**
 * 从 selection 反推当前段落 / 句子 / 单词。
 * 段落定义为最近包含光标的 paragraph 或 heading 节点。
 *
 * splitFn 决定用什么标点切句——英文用 splitEnglishSentences，中文用 splitChineseSentences。
 * sentenceIndex 是文档级的全局索引（跨段落累计），用于跨面板对齐。
 */
export function getWritingContext(editor: Editor, splitFn: SplitFn): WritingContext {
  const { state } = editor
  const { $from } = state.selection
  const { doc } = state

  // 1. 找到最近的段落级节点
  let paragraphDepth = -1
  for (let d = $from.depth; d > 0; d--) {
    const name = $from.node(d).type.name
    if (name === 'paragraph' || name === 'heading') {
      paragraphDepth = d
      break
    }
  }
  if (paragraphDepth === -1) return EMPTY_CONTEXT

  const paragraphNode = $from.node(paragraphDepth)
  const paragraphText = paragraphNode.textContent
  // paragraph 节点在 doc 中的起始位置（NOT $from.start(paragraphDepth)，
  // 后者是段落内容起点，会把本段落本身累加进 baseIdx）
  const paragraphPos = $from.before(paragraphDepth)
  const paragraphStart = $from.start(paragraphDepth)
  const offsetInParagraph = $from.pos - paragraphStart

  // 2. 累计：本段落之前所有段落的总句数
  let baseIdx = 0
  doc.nodesBetween(0, paragraphPos, (node) => {
    if (node.isTextblock) {
      baseIdx += splitFn(node.textContent).length
    }
  })

  // 3. 在段落中找包含光标的句子
  const sentences = splitFn(paragraphText)
  const found = findSentenceInfo(offsetInParagraph, sentences)
  if (!found) {
    return { ...EMPTY_CONTEXT, paragraph: paragraphText }
  }

  // 4. 在句子中找包含光标的单词
  const offsetInSentence = offsetInParagraph - found.info.start
  const wordInfo = findWordAt(found.info.text, offsetInSentence)

  return {
    paragraph: paragraphText,
    paragraphHlStart: found.info.start,
    paragraphHlEnd: found.info.end,
    sentence: found.info.text,
    sentenceHlStart: wordInfo?.start ?? 0,
    sentenceHlEnd: wordInfo?.end ?? 0,
    word: wordInfo?.text ?? '',
    sentenceIndex: baseIdx + found.idx,
  }
}