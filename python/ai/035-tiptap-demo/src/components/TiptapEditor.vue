<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { EditorContent, useEditor } from '@tiptap/vue-3'
import type { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import {
  collectSentences,
  docText,
  findParagraphAt,
  getSentenceIndex,
  splitEnglishSentences,
  topBlockContentRanges,
} from '../composables/useWritingContext'
import {
  createContextHighlightExtension,
  rerenderDecorations,
} from '../composables/highlightContext'
import {
  getSelectionInfo,
  getSentenceAtCursor,
  getWordInfoAt,
  type SelectionInfo,
  type WordPickMode,
} from '../composables/useSelectionKind'
import { useCopyContent } from '../composables/useCopyContent'
import { loadCustomPrompts, useCustomPrompts } from '../composables/useCustomPrompts'
import {
  runCustomPrompt as runCustomPromptApi,
  type UserPrompt,
} from '../services/customPrompts'
import { SCOPE_LABEL, type PromptScope } from '../constants/customPrompts'
import { ApiError } from '../services/http'
import { matchCase } from '../utils/matchCase'
import { escapeHtml } from '../utils/html'
import AiContextMenu from './AiContextMenu.vue'
import type { AiAction } from '../constants/aiActions'

const props = withDefaults(
  defineProps<{
    /** 来自中文侧的高亮：中文句子 i 时，此处把英文第 i 句高亮 */
    highlightedEnIndex: number
    /** 当前英文句子对应的中文原文，给改写 API 当 sourceSentence */
    chineseSentence: string
    /**
     * 改写/润色/语法那一套 AI 能力要不要开。
     * 英译中模式下英文栏是「源文」，没有中文原文可消歧 → 关掉，只留「📖 查词」。
     */
    aiEnabled?: boolean
    /**
     * 是不是源侧。源侧给「清空」，译侧给「复制」——
     * 切方向时两个框的按钮跟着互换。
     */
    isSource?: boolean
    /** 正在翻译的句子（全局句子索引）—— 这些句子显示 loading */
    loadingIndexes?: number[]
    /** 翻译失败的句子（全局句子索引） */
    failedIndexes?: number[]
    /** 能不能编辑。翻译期间会被锁住 */
    editable?: boolean
    /** 每个英文句子对应的中文（按全局句子索引）—— hover 气泡用 */
    chineseSentences?: string[]
  }>(),
  {
    aiEnabled: true,
    isSource: false,
    loadingIndexes: () => [],
    failedIndexes: () => [],
    editable: true,
    chineseSentences: () => [],
  },
)

const html = defineModel<string>('html', { default: '' })
/** 光标所在句子的文档级全局索引（跨面板对齐的唯一依据） */
const sentenceIndex = defineModel<number>('sentenceIndex', { default: -1 })

// ===== 选区 → 上下文菜单 =====
const selection = ref<SelectionInfo | null>(null)
/** 菜单的 viewport 坐标（配合 Teleport + position: fixed）；null 时不渲染 */
const menuPos = ref<{ top: number; left: number } | null>(null)
/** 工具栏按钮点了之后，菜单直接落到哪个面板；null = 停在主菜单 */
const menuAction = ref<AiAction | null>(null)
/** 光标所在句；为空时工具栏那排 AI 按钮禁用 */
const cursorSentence = ref<SelectionInfo | null>(null)

function detectSelection(editor: Editor | null) {
  // 翻译期间编辑器是锁着的 —— 别弹 AI 菜单（那时的译文还没对齐好）
  if (!props.editable) {
    closeMenu()
    cursorSentence.value = null
    return
  }
  cursorSentence.value = getSentenceAtCursor(editor)
  const info = getSelectionInfo(editor)
  // 关掉 AI 能力时（英译中模式），菜单里只剩「📖 查词」—— 只在选中单个单词时才有意义。
  // 其它情况（比如选了短语）菜单里空荡荡的，索性不弹。
  if (!info || (!props.aiEnabled && info.kind !== 'word')) {
    // 只有光标（没选中东西）→ 菜单该关。
    // 但刚点过鼠标的那一下例外：点词会自动换成那个词的候选词，
    // 先关掉、过一会儿再弹出来 —— 中间那半秒看着就是「一闪而过」。
    if (!info && props.aiEnabled && Date.now() - lastClickAt < 500) return
    selection.value = null
    menuPos.value = null
    anchorBox.value = null
    menuAction.value = null
    autoWordKey = ''
    return
  }
  // 手动选中的 → 走主菜单（不要自动接着上次的动作跑）
  menuAction.value = null
  selection.value = info
  anchorBox.value = selectionBox(editor!, info)
  menuPos.value = placeMenu(anchorBox.value, MENU_H_ESTIMATE)
  void syncMenuPosition()
}

/** 选区在 viewport 里的矩形 —— 菜单贴着它摆。 */
interface AnchorBox {
  top: number
  bottom: number
  left: number
  right: number
}

const anchorBox = ref<AnchorBox | null>(null)
/** 菜单锚点元素（Teleport 到 body 的那个容器） */
const menuAnchorRef = ref<HTMLElement | null>(null)
/** 菜单高度估计值 —— 只是用来决定「放上面还是放下面」，真正定位用实测高度。 */
const MENU_H_ESTIMATE = 280

function selectionBox(editor: Editor, info: SelectionInfo): AnchorBox {
  const start = editor.view.coordsAtPos(info.from)
  const end = editor.view.coordsAtPos(info.to)
  return { top: start.top, bottom: end.bottom, left: start.left, right: end.right }
}

/**
 * 菜单展开后高度会变（查词词条、语法结果卡片都很长），
 * 所以定位要按**实测高度**算，不能用估计值 ——
 * 否则菜单底部会跑到视口外被裁掉。
 *
 * 用 ResizeObserver 盯着菜单：面板一展开就重新摆一次。
 */
let menuObserver: ResizeObserver | null = null

async function syncMenuPosition() {
  await nextTick()
  observeMenu()
  applyMeasuredPosition()
}

function observeMenu() {
  menuObserver?.disconnect()
  const el = menuAnchorRef.value?.querySelector('.ai-menu')
  if (!el || typeof ResizeObserver === 'undefined') return
  menuObserver = new ResizeObserver(() => applyMeasuredPosition())
  menuObserver.observe(el)
}

/** 按实测高度重新定位。 */
function applyMeasuredPosition() {
  const box = anchorBox.value
  if (!box) return
  const el = menuAnchorRef.value?.querySelector('.ai-menu')
  const height = el?.getBoundingClientRect().height ?? MENU_H_ESTIMATE
  menuPos.value = placeMenu(box, height)
}

/**
 * 用 ProseMirror 的 coordsAtPos 拿选区的 viewport 矩形，
 * 菜单默认贴在上方；上方放不下就翻到下方，最后再夹进视口。
 * coordsAtPos 返回的就是 viewport 坐标，配合 position: fixed 直接用，
 * 不用再减去任何祖先的 offset（菜单被 Teleport 到了 body 上）。
 */

/** 工具栏按钮走这条：菜单贴着光标，而不是贴着整句。 */
function computeCursorMenuPos(editor: Editor, pos: number) {
  const c = editor.view.coordsAtPos(pos)
  return { top: c.top, bottom: c.bottom, left: c.left, right: c.left }
}

function placeMenu(box: AnchorBox, height: number) {
  // 菜单宽度按最宽的一档估：结果卡片比菜单列表宽
  const MENU_W = 320
  const GAP = 8

  // 上方放得下就放上方，放不下就翻到下方
  let y = box.top - GAP - height
  if (y < 4) y = box.bottom + GAP
  // 再夹进视口（面板展开后可能比预估高）
  y = Math.max(4, Math.min(y, Math.max(4, window.innerHeight - height - 4)))

  // 水平方向夹进视口，别让菜单被裁掉
  const x = Math.max(
    8,
    Math.min((box.left + box.right) / 2 - MENU_W / 2, window.innerWidth - MENU_W - 8),
  )

  return { top: y, left: x }
}

/**
 * 工具栏的常驻按钮：不需要选中整句，直接作用于光标所在句。
 * 复用同一套菜单 / 结果卡片，只是把 autoAction 指到对应面板。
 */
function runOnCursorSentence(action: AiAction) {
  const editorInst = editor.value
  if (!editorInst) return
  const info = getSentenceAtCursor(editorInst)
  if (!info) return
  // 每次都给新对象：菜单里 watch(selection) 会据此重置面板状态
  selection.value = { ...info }
  anchorBox.value = computeCursorMenuPos(editorInst, editorInst.state.selection.head)
  menuPos.value = placeMenu(anchorBox.value, MENU_H_ESTIMATE)
  menuAction.value = action
  void syncMenuPosition()
}

// ===== 光标停在词首 / 点击单词 → 直接给出可替换的候选词 =====

/**
 * 自动弹候选词的去重键（from:to）。
 * 同一个词不重复开菜单；菜单关掉时会清掉，下次再点同一个词还能弹。
 */
let autoWordKey = ''

/** 最近一次鼠标点击的时间 —— 用来分辨「点出来的」和「方向键移过来的」。 */
let lastClickAt = 0

/** 触发防抖：光标连续移动时不至于每个位置都发一次请求。 */
const AUTO_WORD_DELAY = 180
let autoWordTimer: ReturnType<typeof setTimeout> | undefined

function clearAutoWordTimer() {
  if (autoWordTimer) clearTimeout(autoWordTimer)
  autoWordTimer = undefined
}

/**
 * 打开「这个词能换成什么」的候选面板。
 */
function openWordCandidates(info: SelectionInfo) {
  const ed = editor.value
  if (!ed) return
  const key = `${info.from}:${info.to}`
  if (key === autoWordKey) return // 这个词已经开着候选了
  autoWordKey = key

  selection.value = info
  anchorBox.value = selectionBox(ed, info)
  menuPos.value = placeMenu(anchorBox.value, MENU_H_ESTIMATE)
  menuAction.value = 'candidates'
  void syncMenuPosition()
}

/**
 * 排一次「弹候选词」。
 *
 * mode='leftEdge' → 只认「光标正好停在词的最左侧」（词在光标右边）
 * mode='any'      → 点击落在词上就算（点的位置直接用，不用等选区更新）
 */
function scheduleWordCandidates(pos: number, mode: WordPickMode) {
  if (!props.aiEnabled) return
  if (!props.editable) return
  const ed = editor.value
  if (!ed) return

  clearAutoWordTimer()
  autoWordTimer = setTimeout(() => {
    autoWordTimer = undefined
    // 有选区 → 选区菜单已经在管了，别再自动弹候选（会把选区顶掉）
    const { from, to } = ed.state.selection
    if (from !== to) return
    const info = getWordInfoAt(ed, pos, mode)
    if (!info) return
    openWordCandidates(info)
  }, AUTO_WORD_DELAY)
}

// ===== 菜单动作 =====

/** 用新文本替换当前选区；单词/短语替换时保留原大小写 */
function applyReplacement(text: string) {
  const editorInst = editor.value
  const sel = selection.value
  if (!editorInst || !sel) return

  // 自定义指令的「所有正文」→ 整篇替换（按换行拆回段落，别把结构拍平）
  if (customTarget.value?.full) {
    const paras = text
      .split(/\n+/)
      .map((p) => p.trim())
      .filter(Boolean)
      .map((p) => `<p>${escapeHtml(p)}</p>`)
      .join('')
    editorInst.commands.setContent(paras || '<p></p>')
    customTarget.value = null
    return
  }

  const replacement =
    sel.kind === 'word' || sel.kind === 'phrase'
      ? matchCase(sel.text, text)
      : text
  editorInst
    .chain()
    .focus()
    .insertContentAt({ from: sel.from, to: sel.to }, replacement)
    .run()
}

function closeMenu() {
  selection.value = null
  menuPos.value = null
  menuAction.value = null
  autoWordKey = ''
  // 自定义指令那套状态也一起收掉
  customResult.value = null
  customError.value = ''
  customBusy.value = false
  customTarget.value = null
  currentCustomPrompt.value = null
}

// 点击编辑器别处 → 关菜单
// 菜单本身 Teleport 到了 body，点它不会走到这里（它自己的 @mousedown.prevent 会先拦下）
function onShellMousedownCapture(e: MouseEvent) {
  if (!selection.value) return

  // 点在某个词上 → 别关：等下会自动换成那个词的候选词。
  // 关了再弹出来，中间那半秒看着就是「一闪而过」。
  if (props.aiEnabled) {
    const ed = editor.value
    if (!ed) {
      closeMenu()
      return
    }
    const pos = ed.view.posAtCoords({ left: e.clientX, top: e.clientY })?.pos
    if (pos != null && getWordInfoAt(ed, pos, 'any')) return
  }

  closeMenu()
}

onBeforeUnmount(() => {
  menuObserver?.disconnect()
  menuObserver = null
  clearAutoWordTimer()
  closeMenu()
  if (hoverRaf) cancelAnimationFrame(hoverRaf)
  clearHover()
})

// 暴露给父组件：翻译完成后整体替换英文段落
defineExpose({
  setHtml(html: string) {
    editor.value?.commands.setContent(html)
  },

  /**
   * 把译文按段落写回（实时翻译用）。
   *
   * 只替换**变了的那几段**，而且从后往前改 —— 改后面的段落不会影响前面段落的
   * doc 位置。整篇 setContent 的话译文框会闪一下、滚动位置还会跳回顶部。
   * 段落数对不上（用户手动加过段落）就整体重建。
   */
  applyParagraphs(paragraphs: string[], changed: number[]) {
    const ed = editor.value
    if (!ed) return
    const ranges = topBlockContentRanges(ed.state.doc)
    if (ranges.length !== paragraphs.length) {
      ed.commands.setContent(paragraphs.map((t) => `<p>${escapeHtml(t)}</p>`).join(''))
      return
    }
    // 从后往前改 —— 改后面的段落不会影响前面段落的 doc 位置。
    // 用 insertText 纯文本替换（不走 HTML 解析），译文里的 < > & 不会被吃掉。
    const tr = ed.state.tr
    for (const p of [...changed].sort((a, b) => b - a)) {
      const r = ranges[p]
      if (!r) continue
      tr.insertText(paragraphs[p], r.from, r.to)
    }
    ed.view.dispatch(tr)
  },
})

// ===== 跨面板高亮 =====
const highlightRef = ref(props.highlightedEnIndex)
watch(
  () => props.highlightedEnIndex,
  (v) => {
    highlightRef.value = v
    if (editor.value) rerenderDecorations(editor.value)
  },
)

// ===== 翻译中的句子 =====
/** 正在翻的句子（全局索引）—— 给装饰扩展读 */
let loadingSet = new Set<number>()
/** 翻失败的句子（全局索引） */
let failedSet = new Set<number>()

watch(
  () => props.loadingIndexes,
  (v) => {
    loadingSet = new Set(v)
    if (editor.value) rerenderDecorations(editor.value)
  },
)
watch(
  () => props.failedIndexes,
  (v) => {
    failedSet = new Set(v)
    if (editor.value) rerenderDecorations(editor.value)
  },
)

// ===== 翻译期间锁住编辑器 =====
watch(
  () => props.editable,
  (v) => {
    editor.value?.setEditable(v)
  },
)

// ===== 鼠标悬停单词 → 气泡显示这句的中文原文 =====

/** 气泡内容与位置；null = 不显示 */
const hoverTip = ref<{
  top: number
  bottom: number
  left: number
  word: string
  chinese: string
  sentence: string
  below: boolean
} | null>(null)

/** 悬停单词在 doc 里的区间 —— 给装饰扩展读 */
let hoverRange: { from: number; to: number } | null = null

function setHoverRange(r: { from: number; to: number } | null) {
  const same =
    (r === null && hoverRange === null) ||
    (r !== null &&
      hoverRange !== null &&
      r.from === hoverRange.from &&
      r.to === hoverRange.to)
  if (same) return
  hoverRange = r
  if (editor.value) rerenderDecorations(editor.value)
}

function clearHover() {
  setHoverRange(null)
  hoverTip.value = null
}

/** 鼠标移动 —— 一帧算一次，别每个 mousemove 都扫一遍文档 */
let hoverRaf = 0
let hoverMouse: { x: number; y: number } | null = null

function onMouseMove(e: MouseEvent) {
  hoverMouse = { x: e.clientX, y: e.clientY }
  if (hoverRaf) return
  hoverRaf = requestAnimationFrame(() => {
    hoverRaf = 0
    if (hoverMouse) updateHover(hoverMouse.x, hoverMouse.y)
  })
}

function updateHover(x: number, y: number) {
  const ed = editor.value
  if (!ed) return
  // AI 菜单开着的时候不弹气泡 —— 两个浮层叠在一起太乱
  if (selection.value) {
    clearHover()
    return
  }
  const pos = ed.view.posAtCoords({ left: x, top: y })?.pos
  if (pos == null) {
    clearHover()
    return
  }
  const info = getWordInfoAt(ed, pos, 'any')
  if (!info) {
    clearHover()
    return
  }

  // 这个词属于第几句（全局句子索引）→ 对齐表里就是它的中文原文
  const spans = collectSentences(ed.state.doc, splitEnglishSentences)
  const span = spans.find((s) => info.from >= s.from && info.to <= s.to)
  const chinese = span ? (props.chineseSentences[span.index] ?? '') : ''

  setHoverRange({ from: info.from, to: info.to })

  const coords = ed.view.coordsAtPos(info.from)
  // 气泡锚在单词左端（CSS 里 translateX(-50%)），左右各留一半宽度
  const half = 180
  const maxLeft = Math.max(half, window.innerWidth - half)
  const left = Math.min(Math.max(coords.left, half), maxLeft)
  hoverTip.value = {
    top: coords.top,
    bottom: coords.bottom,
    left,
    word: info.text,
    chinese,
    sentence: info.sentence,
    // 顶到屏幕上方就翻到单词下面
    below: coords.top < 90,
  }
}

// ===== 用户自定义指令 =====

/** 当前用户的指令（跟着当前生效的 provider 走） */
const { prompts: customPrompts, toolbarPrompts } = useCustomPrompts()

/** 正在跑的那条指令（「继续调整」时要知道是哪一条） */
const currentCustomPrompt = ref<UserPrompt | null>(null)

/** 这一轮指令要处理的文本范围。 */
interface CustomTarget {
  from: number
  to: number
  /** 要发给模型的正文 */
  text: string
  /** 自动填的上下文（比 text 更大一层） */
  context?: string
  /** 对应的中文原文（有的话带上，消歧用） */
  sourceSentence?: string
  /** 是不是「所有正文」—— 接受时要整篇替换 */
  full: boolean
}

const customTarget = ref<CustomTarget | null>(null)
const customResult = ref<{ original: string; rewritten: string } | null>(null)
const customBusy = ref(false)
const customError = ref('')

/** 截断上下文 —— 全文摘要别把 prompt 撑爆。 */
function truncate(text: string, max: number): string {
  const t = text.trim()
  return t.length > max ? `${t.slice(0, max)}…` : t
}

/**
 * 按作用范围算出「要改哪一段 + 带上什么上下文」。
 *
 *  - 当前句：text = 光标所在句，context = 它所在的段落
 *  - 当前段：text = 光标所在段，context = 全文摘要
 *  - 所有正文：text = 整篇正文，没有 context
 */
function resolveCustomTarget(scope: PromptScope): CustomTarget | null {
  const ed = editor.value
  if (!ed) return null
  const state = ed.state

  if (scope === 'full') {
    const text = docText(state.doc)
    if (!text.trim()) return null
    return { from: 0, to: state.doc.content.size, text, full: true }
  }

  if (scope === 'paragraph') {
    const para = findParagraphAt(state)
    if (!para || !para.text.trim()) return null
    return {
      from: para.from,
      to: para.to,
      text: para.text,
      context: truncate(docText(state.doc), 800),
      full: false,
    }
  }

  // 当前句
  const info = getSentenceAtCursor(ed)
  if (!info || !info.text.trim()) return null
  const para = findParagraphAt(state)
  const spans = collectSentences(state.doc, splitEnglishSentences)
  const span = spans.find((s) => info.from >= s.from && info.to <= s.to)
  return {
    from: info.from,
    to: info.to,
    text: info.text,
    context: para?.text,
    sourceSentence: span ? (props.chineseSentences[span.index] ?? '') : '',
    full: false,
  }
}

/** 把菜单摆到光标处，并让它知道自己处理的是哪一段。 */
function openCustomMenu(target: CustomTarget | null) {
  const ed = editor.value
  if (!ed) return
  const from = target?.from ?? ed.state.selection.from
  const to = target?.to ?? ed.state.selection.to
  selection.value = {
    kind: 'sentence',
    from,
    to,
    text: target?.text ?? '',
    sentence: target?.text ?? '',
    sentenceFrom: from,
    sentenceTo: to,
    paragraph: target?.context ?? '',
    isWholeSentence: false,
  }
  const c = ed.view.coordsAtPos(ed.state.selection.head)
  anchorBox.value = { top: c.top, bottom: c.bottom, left: c.left, right: c.left }
  menuPos.value = placeMenu(anchorBox.value, MENU_H_ESTIMATE)
  void syncMenuPosition()
}

/**
 * 跑一条用户指令。
 *
 * extra = 「继续调整」补的那句话；overrideText = 重新跑时的输入（默认还是原文）。
 */
async function runCustomPrompt(
  prompt: UserPrompt,
  extra = '',
  overrideText?: string,
) {
  const ed = editor.value
  if (!ed) return

  // 「继续调整」要沿用上一轮的范围；新跑一条则按它的作用范围重新取
  const base =
    overrideText !== undefined ? customTarget.value : resolveCustomTarget(prompt.scope)
  if (!base) {
    // 没有可处理的文本 —— 还是把菜单打开，让用户看得到这句话
    customError.value = '这里没有可处理的内容 —— 先把光标放到正文里。'
    customBusy.value = false
    openCustomMenu(null)
    return
  }
  const target: CustomTarget = { ...base, text: overrideText ?? base.text }

  currentCustomPrompt.value = prompt
  customTarget.value = target
  customResult.value = null
  customError.value = ''
  customBusy.value = true
  openCustomMenu(target)

  try {
    const res = await runCustomPromptApi(prompt.id, {
      text: target.text,
      context: target.context,
      sourceSentence: target.sourceSentence,
      ...(extra ? { extraInstruction: extra } : {}),
    })
    customResult.value = { original: target.text, rewritten: res.text }
  } catch (err) {
    customError.value =
      err instanceof ApiError
        ? `[${err.status}] ${err.message}`
        : err instanceof Error
          ? err.message
          : '指令执行失败'
  } finally {
    customBusy.value = false
  }
}

/** 结果卡片里的「继续调整」—— 拿当前草稿再跑一遍同一条指令。 */
function refineCustomPrompt(extra: string) {
  const prompt = currentCustomPrompt.value
  if (!prompt) return
  const draft = customResult.value?.rewritten
  void runCustomPrompt(prompt, extra, draft)
}

// ===== editor 实例（默认空文档） =====

/**
 * 工具栏上的整句操作，收在一个下拉里。
 *
 * 原来 6 个按钮直接摊在工具栏上，把半栏宽度占满了；
 * 现在只留一个「✨ AI」按钮，点开才是完整清单。
 */
const AI_BUTTONS: { action: AiAction; icon: string }[] = [
  { action: 'tone', icon: '🎨' },
  { action: 'improve', icon: '✏️' },
  { action: 'shorten', icon: '✂️' },
  { action: 'expand', icon: '➕' },
  { action: 'grammar', icon: '🔍' },
  { action: 'instruction', icon: '💬' },
]

/** 下拉里的文案。 */
const AI_TITLES: Record<AiAction, string> = {
  candidates: '替换选区',
  tone: '换个语气',
  improve: '润色',
  shorten: '精简',
  expand: '扩写',
  grammar: '检查语法',
  instruction: '自定义指令',
  lookup: '查词',
}

// ===== AI 下拉 =====
/** 下拉菜单宽度 / 高度（高度按项数估 —— 「我的指令」是动态的） */
const AI_DROPDOWN_W = 176
const AI_DROPDOWN_ITEM_H = 34

const aiMenuOpen = ref(false)
/** 下拉在 viewport 里的位置（Teleport 到 body，用 fixed 定位） */
const aiMenuPos = ref<{ top: number; left: number } | null>(null)
const aiBtnRef = ref<HTMLElement | null>(null)
const aiMenuRef = ref<HTMLElement | null>(null)

/** 下拉的实际高度（含「我的指令」那几行）。 */
function aiDropdownHeight(): number {
  const base = AI_BUTTONS.length * AI_DROPDOWN_ITEM_H + 16
  const extra = customPrompts.value.length ? customPrompts.value.length * AI_DROPDOWN_ITEM_H + 28 : 0
  return base + extra
}

function toggleAiMenu() {
  if (aiMenuOpen.value) {
    aiMenuOpen.value = false
    return
  }
  const el = aiBtnRef.value
  if (!el) return
  const rect = el.getBoundingClientRect()
  // 右对齐按钮；下方放不下就翻到上面
  const left = Math.max(
    8,
    Math.min(rect.right - AI_DROPDOWN_W, window.innerWidth - AI_DROPDOWN_W - 8),
  )
  const below = rect.bottom + 6
  const height = aiDropdownHeight()
  const top =
    below + height > window.innerHeight
      ? Math.max(8, rect.top - height - 6)
      : below
  aiMenuPos.value = { top, left }
  aiMenuOpen.value = true
}

/** 点外面 / 选中一项就收起。 */
function closeAiMenu() {
  aiMenuOpen.value = false
}

/** 文档级点击：点在按钮和菜单外面就收起。 */
function onDocMouseDown(e: MouseEvent) {
  if (!aiMenuOpen.value) return
  const t = e.target as Node | null
  if (t && (aiBtnRef.value?.contains(t) || aiMenuRef.value?.contains(t))) return
  closeAiMenu()
}

/** Esc 收起下拉。 */
function onKeydown(e: KeyboardEvent) {
  if (e.key !== 'Escape' || !aiMenuOpen.value) return
  e.stopPropagation()
  closeAiMenu()
}

onMounted(() => {
  document.addEventListener('mousedown', onDocMouseDown)
  window.addEventListener('keydown', onKeydown)
  // 页面滚动 → 气泡是 viewport 坐标，会跑位
  window.addEventListener('scroll', clearHover)
  // 拉一次当前用户的自定义指令（跟着当前生效的 provider）
  // 强制刷新 —— 管理员在设置页换过 provider 之后回来，看到的得是新那套
  void loadCustomPrompts(true)
})
onBeforeUnmount(() => {
  document.removeEventListener('mousedown', onDocMouseDown)
  window.removeEventListener('keydown', onKeydown)
  window.removeEventListener('scroll', clearHover)
})

/** 下拉里点一项 → 跑那个动作，然后收起。 */
function runAiAction(action: AiAction) {
  closeAiMenu()
  runOnCursorSentence(action)
}

const editor = useEditor({
  content: '<p></p>',
  extensions: [
    StarterKit,
    createContextHighlightExtension({
      splitFn: splitEnglishSentences,
      getCrossIndex: () => highlightRef.value,
      showContextLayers: () => true,
      getLoadingIndexes: () => loadingSet,
      getFailedIndexes: () => failedSet,
      getHoverRange: () => hoverRange,
    }),
  ],
  onUpdate({ editor }) {
    html.value = editor.getHTML()
    sentenceIndex.value = getSentenceIndex(editor.state, splitEnglishSentences)
    detectSelection(editor)
    // 内容变了 → 气泡里的中文原文可能已经对不上，收掉
    clearHover()
  },
  editorProps: {
    // 点在某个词上 → 直接给出这个词能换成什么（不用先拖选）
    //
    // ProseMirror 的 handleClick 跑在**选区更新之前**（mouseup 里先调它，
    // 再更新选区），所以这里直接用点击给的 pos，别等选区。
    handleClick: (_view, pos) => {
      lastClickAt = Date.now()
      scheduleWordCandidates(pos, 'any')
      return false
    },
    handleDOMEvents: {
      // 鼠标停在单词上 → 气泡显示这句的中文原文
      mousemove: (_view, event) => {
        const e = event as MouseEvent
        onMouseMove(e)
        return false
      },
    },
  },
  onSelectionUpdate({ editor, transaction }) {
    sentenceIndex.value = getSentenceIndex(editor.state, splitEnglishSentences)
    detectSelection(editor)
    // 打字会同时改文档 —— 那不是「光标停在词首」，别自动弹候选
    if (transaction.docChanged) return
    const { from, to } = editor.state.selection
    if (from !== to) return // 有选区 → 走选区菜单
    // 刚点过鼠标 → 点击那一路已经排过了，别用「词首」规则把它顶掉
    if (Date.now() - lastClickAt < 500) return
    scheduleWordCandidates(from, 'leftEdge')
  },
  onCreate({ editor }) {
    editor.setEditable(props.editable)
    html.value = editor.getHTML()
    sentenceIndex.value = getSentenceIndex(editor.state, splitEnglishSentences)
    detectSelection(editor)
  },
})

// ===== 复制内容 =====
const { copied, copyContent } = useCopyContent(() => editor.value)
</script>

<template>
  <div
    class="editor-shell en-shell"
    :class="{ 'is-locked': !editable }"
    @mousedown.capture="onShellMousedownCapture"
    @mouseleave="clearHover"
    @scroll.capture="clearHover"
  >
    <div v-if="editor" class="toolbar">
      <!-- 文本格式 -->
      <div class="toolbar__group">
        <button type="button" :class="{ active: editor.isActive('bold') }"
          :disabled="!editable || !editor.can().chain().focus().toggleBold().run()"
          @click="editor.chain().focus().toggleBold().run()" title="加粗 (Ctrl+B)">
          <strong>B</strong>
        </button>
        <button type="button" :class="{ active: editor.isActive('italic') }"
          :disabled="!editable || !editor.can().chain().focus().toggleItalic().run()"
          @click="editor.chain().focus().toggleItalic().run()" title="斜体 (Ctrl+I)">
          <em>I</em>
        </button>
        <button type="button" :class="{ active: editor.isActive('strike') }"
          :disabled="!editable || !editor.can().chain().focus().toggleStrike().run()"
          @click="editor.chain().focus().toggleStrike().run()" title="删除线">
          <s>S</s>
        </button>
        <button type="button" :class="{ active: editor.isActive('underline') }"
          :disabled="!editable || !editor.can().chain().focus().toggleUnderline().run()"
          @click="editor.chain().focus().toggleUnderline().run()" title="下划线 (Ctrl+U)">
          <u>U</u>
        </button>
      </div>

      <!-- 历史 -->
      <div class="toolbar__group">
        <button type="button" :disabled="!editable || !editor.can().chain().focus().undo().run()"
          @click="editor.chain().focus().undo().run()" title="撤销 (Ctrl+Z)">↶</button>
        <button type="button" :disabled="!editable || !editor.can().chain().focus().redo().run()"
          @click="editor.chain().focus().redo().run()" title="重做 (Ctrl+Shift+Z)">↷</button>
      </div>

      <!-- 用户自定义指令：钉在工具栏上的那些 -->
      <div v-if="toolbarPrompts.length" class="toolbar__group toolbar__group--prompts">
        <button
          v-for="p in toolbarPrompts"
          :key="p.id"
          type="button"
          class="toolbar__btn-prompt"
          :disabled="!editable"
          :title="`${p.label}（${SCOPE_LABEL[p.scope]}）`"
          @click="runCustomPrompt(p)"
        >
          <span class="toolbar__btn-prompt-icon">{{ p.icon }}</span>
          <span class="toolbar__btn-prompt-label">{{ p.label }}</span>
        </button>
      </div>

      <!-- 源侧：清空；译侧：复制 -->
      <div class="toolbar__group" :class="{ 'toolbar__group--end': !aiEnabled }">
        <button
          v-if="isSource"
          type="button"
          class="toolbar__btn-clear"
          title="清空英文内容"
          :disabled="!editable || !editor.getText().trim()"
          @click="editor.commands.clearContent()"
        >
          清空
        </button>
        <button
          v-else
          type="button"
          class="toolbar__btn-copy"
          :class="{ 'is-copied': copied }"
          :disabled="!editor.getText().trim()"
          :title="copied ? '已复制到剪贴板' : '复制英文内容'"
          @click="copyContent"
        >
          {{ copied ? '✓ 已复制' : '复制' }}
        </button>
      </div>

      <!-- 整句操作：收在一个下拉里，作用于光标所在句，不需要选中 -->
      <div v-if="aiEnabled || customPrompts.length" class="toolbar__group toolbar__group--ai">
        <button
          ref="aiBtnRef"
          type="button"
          class="toolbar__btn-ai-menu"
          :class="{ active: aiMenuOpen }"
          :aria-expanded="aiMenuOpen"
          aria-haspopup="menu"
          :disabled="!editable"
          title="AI 指令（作用于光标所在句）"
          @click="toggleAiMenu"
        >
          ✨ AI ▾
        </button>
      </div>
    </div>

    <EditorContent :editor="editor" />

    <!-- AI 下拉：Teleport 到 body（编辑器外框 overflow:hidden 会裁掉它） -->
    <Teleport to="body">
      <div
        v-if="aiMenuOpen && aiMenuPos"
        ref="aiMenuRef"
        class="ai-dropdown"
        role="menu"
        :style="{ top: `${aiMenuPos.top}px`, left: `${aiMenuPos.left}px` }"
      >
        <template v-if="aiEnabled">
          <button
            v-for="a in AI_BUTTONS"
            :key="a.action"
            type="button"
            role="menuitem"
            class="ai-dropdown__item"
            :disabled="!cursorSentence"
            :title="`对光标所在句：${AI_TITLES[a.action]}`"
            @click="runAiAction(a.action)"
          >
            <span class="ai-dropdown__icon">{{ a.icon }}</span>
            <span>{{ AI_TITLES[a.action] }}</span>
          </button>
        </template>

        <!-- 我的指令：用户自己写的那些 -->
        <template v-if="customPrompts.length">
          <div class="ai-dropdown__sep" :class="{ 'ai-dropdown__sep--first': !aiEnabled }">
            我的指令
          </div>
          <button
            v-for="p in customPrompts"
            :key="p.id"
            type="button"
            role="menuitem"
            class="ai-dropdown__item ai-dropdown__item--prompt"
            :disabled="!editable"
            :title="`${p.instruction}（作用于${SCOPE_LABEL[p.scope]}）`"
            @click="runCustomPrompt(p)"
          >
            <span class="ai-dropdown__icon">{{ p.icon }}</span>
            <span>{{ p.label }}</span>
            <span class="ai-dropdown__scope">{{ SCOPE_LABEL[p.scope] }}</span>
          </button>
        </template>
      </div>
    </Teleport>

    <!-- 选区上下文菜单：选中单词/短语/整句时自动弹出 -->
    <Teleport to="body">
      <div
        v-if="selection && menuPos"
        ref="menuAnchorRef"
        class="ai-menu-anchor"
        :style="{ top: `${menuPos.top}px`, left: `${menuPos.left}px` }"
      >
        <AiContextMenu
          :selection="selection"
          :chinese-sentence="chineseSentence"
          :auto-action="menuAction"
          :allow-ai="aiEnabled"
          :custom-result="customResult"
          :custom-busy="customBusy"
          :custom-error="customError"
          @replace="applyReplacement"
          @refine="refineCustomPrompt"
          @close="closeMenu"
        />
      </div>
    </Teleport>

    <!-- 悬停气泡：鼠标停在单词上 → 显示这句的中文原文 -->
    <Teleport to="body">
      <div
        v-if="hoverTip"
        class="hover-tip"
        :style="{
          left: `${hoverTip.left}px`,
          top: `${hoverTip.below ? hoverTip.bottom : hoverTip.top}px`,
          transform: hoverTip.below ? 'translate(-50%, 0)' : 'translate(-50%, -100%)',
        }"
      >
        <div class="hover-tip__word">{{ hoverTip.word }}</div>
        <div v-if="hoverTip.chinese" class="hover-tip__zh">{{ hoverTip.chinese }}</div>
        <div v-else class="hover-tip__zh hover-tip__zh--empty">{{ hoverTip.sentence }}</div>
      </div>
    </Teleport>
  </div>
</template>
