import type { Editor } from '@tiptap/core'
import { splitEnglishSentences, type Sentence } from './useWritingContext'

/**
 * 选区分类 —— 上下文菜单弹什么，取决于选中了什么。
 *
 *   光标（无选区）  → null，不弹
 *   跨句 / 跨段      → null，不弹（需求：大于一句就不给上下文菜单）
 *   单个单词        → 'word'
 *   整句            → 'sentence'
 *   其它（单句内的多词）→ 'phrase'
 */

export type SelectionKind = 'word' | 'phrase' | 'sentence'

export interface SelectionInfo {
  kind: SelectionKind
  /** 选区在 doc 中的绝对位置，用于 splice 替换 */
  from: number
  to: number
  /** 选区文本（trim 过） */
  text: string
  /** 选区所在整句的文本，用于 grammar-check / sentence 级改写 */
  sentence: string
  /** 该句在 doc 中的绝对位置区间，用于标点/语法问题的定位基准 */
  sentenceFrom: number
  sentenceTo: number
  /** 选区所在段落文本，传给 API 当上下文 */
  paragraph: string
  /** 选区是否正好是整句 */
  isWholeSentence: boolean
}

/**
 * 找到 $from 所在的 paragraph / heading 节点。
 * 段落定位逻辑与 useWritingContext.getWritingContext 一致。
 */
function findTextblock(editor: Editor): { text: string; start: number } | null {
  const { $from } = editor.state.selection
  for (let d = $from.depth; d > 0; d--) {
    const name = $from.node(d).type.name
    if (name === 'paragraph' || name === 'heading') {
      return {
        text: $from.node(d).textContent,
        // start(depth) 是内容起点；before(depth) 是节点起点，会差一个
        start: $from.start(d),
      }
    }
  }
  return null
}

/**
 * 选区落在哪一句里。
 *
 * splitEnglishSentences 切出来的偏移是相对段落文本的，加 paragraphStart 才回到 doc 坐标。
 */
function locateSentence(
  paragraphStart: number,
  paragraphText: string,
  from: number,
  to: number,
): { sentence: Sentence; sentenceFrom: number; sentenceTo: number } | null {
  const sentences = splitEnglishSentences(paragraphText)
  const fromOffset = from - paragraphStart
  const toOffset = to - paragraphStart

  for (const s of sentences) {
    // 选区必须整体落在同一句内；to 允许恰好等于句尾
    if (fromOffset >= s.start && toOffset <= s.end) {
      return {
        sentence: s,
        sentenceFrom: paragraphStart + s.start,
        sentenceTo: paragraphStart + s.end,
      }
    }
  }
  return null
}

/**
 * 分类当前选区。返回 null 表示「不该弹菜单」。
 */
export function getSelectionInfo(editor: Editor | null): SelectionInfo | null {
  if (!editor) return null

  const { from, to } = editor.state.selection
  if (from === to) return null // 只有光标

  const text = editor.state.doc.textBetween(from, to, '', '')
  if (!text.trim()) return null

  const block = findTextblock(editor)
  if (!block) return null

  const located = locateSentence(block.start, block.text, from, to)
  if (!located) return null // 跨句或跨段 → 不弹

  // 选中文本 trim 后与整句 trim 后一致 → 用户选的就是整句
  const isWholeSentence = text.trim() === located.sentence.text.trim()

  let kind: SelectionKind
  if (isWholeSentence) {
    kind = 'sentence'
  } else if (/^\w+$/.test(text.trim())) {
    kind = 'word'
  } else {
    kind = 'phrase'
  }

  return {
    kind,
    from,
    to,
    text: text.trim(),
    sentence: located.sentence.text,
    sentenceFrom: located.sentenceFrom,
    sentenceTo: located.sentenceTo,
    paragraph: block.text.slice(0, 400),
    isWholeSentence,
  }
}

/**
 * 光标所在的整句，不需要选中任何字符。
 * 给工具栏那排常驻按钮用——它们永远作用于「光标所在句」，跟有没有选区无关。
 *
 * 光标落在句间空白（两个句号之间）时取最近的那一句：向前找第一个 start <= pos 的句子，
 * 而不是直接返回 null——光标在句子中间换行到末尾时也属于这种情况。
 */
export function getSentenceAtCursor(editor: Editor | null): SelectionInfo | null {
  if (!editor) return null

  const block = findTextblock(editor)
  if (!block || !block.text.trim()) return null

  const pos = editor.state.selection.from
  const offset = Math.min(Math.max(pos - block.start, 0), block.text.length)

  const sentences = splitEnglishSentences(block.text)
  if (sentences.length === 0) return null

  let hit = sentences.find((s) => offset >= s.start && offset <= s.end)
  if (!hit) {
    // 光标在句子之间的空白里 → 归到前面那句
    hit = [...sentences].reverse().find((s) => s.start <= offset)
    if (!hit) hit = sentences[0]
  }

  return {
    kind: 'sentence',
    from: block.start + hit.start,
    to: block.start + hit.end,
    text: hit.text,
    sentence: hit.text,
    sentenceFrom: block.start + hit.start,
    sentenceTo: block.start + hit.end,
    paragraph: block.text.slice(0, 400),
    isWholeSentence: true,
  }
}
