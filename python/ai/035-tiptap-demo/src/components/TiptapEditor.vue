<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { EditorContent, useEditor } from '@tiptap/vue-3'
import type { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import {
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
  type SelectionInfo,
} from '../composables/useSelectionKind'
import { matchCase } from '../utils/matchCase'
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
  }>(),
  { aiEnabled: true },
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
  cursorSentence.value = getSentenceAtCursor(editor)
  const info = getSelectionInfo(editor)
  // 关掉 AI 能力时（英译中模式），菜单里只剩「📖 查词」—— 只在选中单个单词时才有意义。
  // 其它情况（比如选了短语）菜单里空荡荡的，索性不弹。
  if (!info || (!props.aiEnabled && info.kind !== 'word')) {
    selection.value = null
    menuPos.value = null
    anchorBox.value = null
    return
  }
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

// ===== 菜单动作 =====

/** 用新文本替换当前选区；单词/短语替换时保留原大小写 */
function applyReplacement(text: string) {
  const editorInst = editor.value
  const sel = selection.value
  if (!editorInst || !sel) return
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
}

// 点击编辑器别处 → 关菜单
// 菜单本身 Teleport 到了 body，点它不会走到这里（它自己的 @mousedown.prevent 会先拦下）
function onShellMousedownCapture() {
  if (!selection.value) return
  closeMenu()
}

onBeforeUnmount(() => {
  menuObserver?.disconnect()
  menuObserver = null
  closeMenu()
  if (copiedTimer) clearTimeout(copiedTimer)
})

// ===== 复制内容 =====
/** 刚复制完（按钮变个色提示一下） */
const copied = ref(false)
let copiedTimer: ReturnType<typeof setTimeout> | undefined

/**
 * 把英文框里的内容复制成纯文本。
 *
 * 段落之间用换行，段内的硬换行（Shift+Enter）也还原成换行 ——
 * 直接用 getText() 的话硬换行会被吃掉，两行粘成一行。
 */
async function copyContent() {
  const ed = editor.value
  if (!ed) return
  const doc = ed.state.doc
  const text = doc.textBetween(
    0,
    doc.content.size,
    '\n',
    (node) => (node.type.name === 'hardBreak' ? '\n' : ''),
  )
  if (!text.trim()) return

  try {
    await navigator.clipboard.writeText(text)
  } catch {
    // 剪贴板 API 不可用（非安全上下文，比如用局域网 IP 访问）→ 退回老办法
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    document.execCommand('copy')
    document.body.removeChild(ta)
  }

  copied.value = true
  if (copiedTimer) clearTimeout(copiedTimer)
  copiedTimer = setTimeout(() => {
    copied.value = false
  }, 1500)
}

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
      ed.commands.setContent(paragraphs.map((t) => `<p>${t}</p>`).join(''))
      return
    }
    for (const p of [...changed].sort((a, b) => b - a)) {
      const r = ranges[p]
      if (!r) continue
      ed.commands.insertContentAt({ from: r.from, to: r.to }, paragraphs[p])
    }
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
/** 下拉菜单宽度 / 高度（项数固定，高度可预估） */
const AI_DROPDOWN_W = 176
const AI_DROPDOWN_H = 240

const aiMenuOpen = ref(false)
/** 下拉在 viewport 里的位置（Teleport 到 body，用 fixed 定位） */
const aiMenuPos = ref<{ top: number; left: number } | null>(null)
const aiBtnRef = ref<HTMLElement | null>(null)
const aiMenuRef = ref<HTMLElement | null>(null)

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
  const top =
    below + AI_DROPDOWN_H > window.innerHeight
      ? Math.max(8, rect.top - AI_DROPDOWN_H - 6)
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
})
onBeforeUnmount(() => {
  document.removeEventListener('mousedown', onDocMouseDown)
  window.removeEventListener('keydown', onKeydown)
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
    }),
  ],
  onUpdate({ editor }) {
    html.value = editor.getHTML()
    sentenceIndex.value = getSentenceIndex(editor.state, splitEnglishSentences)
    detectSelection(editor)
  },
  onSelectionUpdate({ editor }) {
    sentenceIndex.value = getSentenceIndex(editor.state, splitEnglishSentences)
    detectSelection(editor)
  },
  onCreate({ editor }) {
    html.value = editor.getHTML()
    sentenceIndex.value = getSentenceIndex(editor.state, splitEnglishSentences)
    detectSelection(editor)
  },
})
</script>

<template>
  <div class="editor-shell en-shell" @mousedown.capture="onShellMousedownCapture">
    <div v-if="editor" class="toolbar">
      <!-- 文本格式 -->
      <div class="toolbar__group">
        <button type="button" :class="{ active: editor.isActive('bold') }"
          :disabled="!editor.can().chain().focus().toggleBold().run()"
          @click="editor.chain().focus().toggleBold().run()" title="加粗 (Ctrl+B)">
          <strong>B</strong>
        </button>
        <button type="button" :class="{ active: editor.isActive('italic') }"
          :disabled="!editor.can().chain().focus().toggleItalic().run()"
          @click="editor.chain().focus().toggleItalic().run()" title="斜体 (Ctrl+I)">
          <em>I</em>
        </button>
        <button type="button" :class="{ active: editor.isActive('strike') }"
          :disabled="!editor.can().chain().focus().toggleStrike().run()"
          @click="editor.chain().focus().toggleStrike().run()" title="删除线">
          <s>S</s>
        </button>
        <button type="button" :class="{ active: editor.isActive('underline') }"
          :disabled="!editor.can().chain().focus().toggleUnderline().run()"
          @click="editor.chain().focus().toggleUnderline().run()" title="下划线 (Ctrl+U)">
          <u>U</u>
        </button>
      </div>

      <!-- 历史 -->
      <div class="toolbar__group">
        <button type="button" :disabled="!editor.can().chain().focus().undo().run()"
          @click="editor.chain().focus().undo().run()" title="撤销 (Ctrl+Z)">↶</button>
        <button type="button" :disabled="!editor.can().chain().focus().redo().run()"
          @click="editor.chain().focus().redo().run()" title="重做 (Ctrl+Shift+Z)">↷</button>
      </div>

      <!-- 复制内容 -->
      <div class="toolbar__group">
        <button
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
      <div v-if="aiEnabled" class="toolbar__group toolbar__group--ai">
        <button
          ref="aiBtnRef"
          type="button"
          class="toolbar__btn-ai-menu"
          :class="{ active: aiMenuOpen }"
          :aria-expanded="aiMenuOpen"
          aria-haspopup="menu"
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
          @replace="applyReplacement"
          @close="closeMenu"
        />
      </div>
    </Teleport>
  </div>
</template>
