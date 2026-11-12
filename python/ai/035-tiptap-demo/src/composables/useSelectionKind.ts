import type { Editor } from '@tiptap/core'
import {
  buildChunks,
  chunkText,
  posAt,
  splitEnglishSentences,
  textOffsetAt,
  type Chunk,
  type Sentence,
} from './useWritingContext'

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
 * 段落定位逻辑与 useWritingContext.resolveContext 一致。
 *
 * 偏移和 doc 位置不是一回事（hardBreak 在 textContent 里不占字符，在 doc 里占 1 个位置），
 * 所以这里带上 chunks，用 posAt / textOffsetAt 换算。
 */
function findTextblock(
  editor: Editor,
): { text: string; chunks: Chunk[] } | null {
  return findTextblockAt(editor, editor.state.selection.from)
}

/**
 * 找 pos 所在的 paragraph / heading 节点。
 *
 * 和 findTextblock 一样，只是位置由参数给 —— 鼠标点的位置不一定等于选区位置。
 */
function findTextblockAt(
  editor: Editor,
  pos: number,
): { text: string; chunks: Chunk[] } | null {
  const doc = editor.state.doc
  const $pos = doc.resolve(Math.min(Math.max(pos, 0), doc.content.size))
  for (let d = $pos.depth; d > 0; d--) {
    const name = $pos.node(d).type.name
    if (name === 'paragraph' || name === 'heading') {
      // before(depth) 是节点起点（段落内容从 +1 开始）
      const chunks = buildChunks($pos.node(d), $pos.before(d))
      return {
        text: chunkText(chunks),
        chunks,
      }
    }
  }
  return null
}

/**
 * 选区落在哪一句里。
 *
 * splitEnglishSentences 切出来的偏移是相对段落文本的，
 * 要用 posAt 换算回 doc 坐标（不能直接加段落起点 —— 硬换行会偏）。
 */
function locateSentence(
  block: { chunks: Chunk[]; text: string },
  from: number,
  to: number,
): { sentence: Sentence; sentenceFrom: number; sentenceTo: number } | null {
  const sentences = splitEnglishSentences(block.text)
  const fromOffset = textOffsetAt(block.chunks, from)
  const toOffset = textOffsetAt(block.chunks, to)

  for (const s of sentences) {
    // 选区必须整体落在同一句内；to 允许恰好等于句尾
    if (fromOffset >= s.start && toOffset <= s.end) {
      return {
        sentence: s,
        sentenceFrom: posAt(block.chunks, s.start, 'start'),
        sentenceTo: posAt(block.chunks, s.end, 'end'),
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

  const located = locateSentence(block, from, to)
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

// ===== 光标 / 点击处的单词 =====

/** 词字符：字母 / 数字 / 下划线（和上面 /\w+$/ 的判定保持一致）。 */
function isWordChar(c: string): boolean {
  return /[A-Za-z0-9_]/.test(c)
}

/** 撇号 —— don't 里的那个，词**内部**的撇号算一个词的一部分。 */
function isApostrophe(c: string): boolean {
  return c === "'" || c === '\u2019'
}

/** 取词的方式 */
export type WordPickMode = 'leftEdge' | 'any'

/**
 * 从 offset 处找词。
 *
 * mode='leftEdge' → 光标必须正好停在词的**最左侧**（词在光标右边）；
 *                    光标在词中间（方向键走进这个词）不算。
 * mode='any'      → 点在词上就算：光标在词里取整个词；
 *                    光标停在词的**末尾**（点在词的右半边）取左边那个词。
 */
function scanWord(
  text: string,
  offset: number,
  mode: WordPickMode,
): { from: number; to: number } | null {
  if (offset < 0 || offset > text.length) return null

  let s = offset
  let e = offset

  if (isWordChar(text[offset] ?? '')) {
    // 向左扩
    while (s > 0) {
      const c = text[s - 1]
      if (isWordChar(c)) {
        s -= 1
        continue
      }
      // 词内部的撇号：don't 的 n 前面那个 ' 算词的一部分
      if (isApostrophe(c) && s >= 2 && isWordChar(text[s - 2])) {
        s -= 2
        continue
      }
      break
    }

    // 向右扩
    while (e < text.length) {
      const c = text[e]
      if (isWordChar(c)) {
        e += 1
        continue
      }
      if (isApostrophe(c) && e + 1 < text.length && isWordChar(text[e + 1])) {
        e += 2
        continue
      }
      break
    }
  } else if (mode === 'any' && offset > 0 && isWordChar(text[offset - 1])) {
    // 光标停在词的末尾（点在词的右半边）→ 取左边那个词
    e = offset
    s = offset - 1
    while (s > 0) {
      const c = text[s - 1]
      if (isWordChar(c)) {
        s -= 1
        continue
      }
      if (isApostrophe(c) && s >= 2 && isWordChar(text[s - 2])) {
        s -= 2
        continue
      }
      break
    }
  } else {
    return null
  }

  if (mode === 'leftEdge' && s !== offset) return null
  return { from: s, to: e }
}

/**
 * 光标 / 点击处的单词，组装成 SelectionInfo（和选区那套同一个结构）。
 *
 * 两种触发：
 *   mode='leftEdge' → 光标停在词的最左侧（词在光标右边）
 *   mode='any'      → 鼠标点在这个词上（点词中间、点词的右半边都算）
 */
export function getWordInfoAt(
  editor: Editor | null,
  pos: number,
  mode: WordPickMode,
): SelectionInfo | null {
  if (!editor) return null

  const block = findTextblockAt(editor, pos)
  if (!block) return null

  const offset = textOffsetAt(block.chunks, pos)
  const word = scanWord(block.text, offset, mode)
  if (!word) return null

  const from = posAt(block.chunks, word.from, 'start')
  const to = posAt(block.chunks, word.to, 'end')
  const text = block.text.slice(word.from, word.to).trim()
  if (!text) return null

  const located = locateSentence(block, from, to)
  if (!located) return null

  // 和选区分类同一套判定：纯单词 → 'word'，带撇号的（don't）算 'phrase'
  const kind: SelectionKind = /^\w+$/.test(text) ? 'word' : 'phrase'

  return {
    kind,
    from,
    to,
    text,
    sentence: located.sentence.text,
    sentenceFrom: located.sentenceFrom,
    sentenceTo: located.sentenceTo,
    paragraph: block.text.slice(0, 400),
    isWholeSentence: false,
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
  // doc 位置 → 段落纯文本里的偏移（硬换行会差位，所以不能直接减段落起点）
  const offset = Math.min(Math.max(textOffsetAt(block.chunks, pos), 0), block.text.length)

  const sentences = splitEnglishSentences(block.text)
  if (sentences.length === 0) return null

  let hit = sentences.find((s) => offset >= s.start && offset <= s.end)
  if (!hit) {
    // 光标在句子之间的空白里 → 归到前面那句
    hit = [...sentences].reverse().find((s) => s.start <= offset)
    if (!hit) hit = sentences[0]
  }

  const from = posAt(block.chunks, hit.start, 'start')
  const to = posAt(block.chunks, hit.end, 'end')

  return {
    kind: 'sentence',
    from,
    to,
    text: hit.text,
    sentence: hit.text,
    sentenceFrom: from,
    sentenceTo: to,
    paragraph: block.text.slice(0, 400),
    isWholeSentence: true,
  }
}
