import type { Editor } from '@tiptap/core'

/**
 * 当前光标所在的写作上下文。
 * - paragraph: 整个段落文本
 * - paragraphHlStart/End: 当前句子在段落中的起止偏移（用于高亮）
 * - sentence: 当前句子文本
 * - sentenceHlStart/End: 当前单词在句子中的起止偏移
 * - word: 当前光标所在的单词（去尾随标点前的部分）
 */
export interface WritingContext {
  paragraph: string
  paragraphHlStart: number
  paragraphHlEnd: number
  sentence: string
  sentenceHlStart: number
  sentenceHlEnd: number
  word: string
}

export const EMPTY_CONTEXT: WritingContext = {
  paragraph: '',
  paragraphHlStart: 0,
  paragraphHlEnd: 0,
  sentence: '',
  sentenceHlStart: 0,
  sentenceHlEnd: 0,
  word: '',
}

/**
 * 在 text 中定位包含 offset 的"句子"。
 * 句子 = 连续的非句末标点字符 + 可选的句末标点 (. ! ?)。
 * 段落内部的 hardBreak 视为句子边界（用 \n 区分）。
 */
function findSentenceAt(
  text: string,
  offset: number,
): { text: string; start: number; end: number } | null {
  const re = /[^.!?\n]+[.!?]?/g
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

/**
 * 从当前 selection 反推光标所在的段落 / 句子 / 单词。
 * 段落定义为最近包含光标的 paragraph 或 heading 节点。
 */
export function getWritingContext(editor: Editor): WritingContext {
  const { $from } = editor.state.selection

  // 1. 找到最近的段落级节点（paragraph / heading）
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
  const paragraphStart = $from.start(paragraphDepth) // 段落节点内容在 doc 中的起始位置
  const offsetInParagraph = $from.pos - paragraphStart

  // 2. 在段落中找包含光标的句子
  const sentenceInfo = findSentenceAt(paragraphText, offsetInParagraph)
  if (!sentenceInfo) {
    return {
      ...EMPTY_CONTEXT,
      paragraph: paragraphText,
    }
  }

  // 3. 在句子中找包含光标的单词
  const offsetInSentence = offsetInParagraph - sentenceInfo.start
  const wordInfo = findWordAt(sentenceInfo.text, offsetInSentence)

  return {
    paragraph: paragraphText,
    paragraphHlStart: sentenceInfo.start,
    paragraphHlEnd: sentenceInfo.end,
    sentence: sentenceInfo.text,
    sentenceHlStart: wordInfo?.start ?? 0,
    sentenceHlEnd: wordInfo?.end ?? 0,
    word: wordInfo?.text ?? '',
  }
}