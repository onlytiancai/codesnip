<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import { EditorContent, useEditor } from '@tiptap/vue-3'
import type { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import {
  EMPTY_CONTEXT,
  getWritingContext,
  splitEnglishSentences,
  type WritingContext,
} from '../composables/useWritingContext'
import {
  createHighlightSentenceExtension,
  rerenderDecorations,
} from '../composables/highlightSentence'
import {
  getSelectionInfo,
  getSentenceAtCursor,
  type SelectionInfo,
} from '../composables/useSelectionKind'
import { matchCase } from '../utils/matchCase'
import AiContextMenu from './AiContextMenu.vue'
import type { AiAction } from '../constants/aiActions'

const props = defineProps<{
  /** 来自中文侧的高亮：中文句子 i 时，此处把英文第 i 句高亮 */
  highlightedEnIndex: number
  /** 当前英文句子对应的中文原文，给改写 API 当 sourceSentence */
  chineseSentence: string
}>()

const html = defineModel<string>('html', { default: '' })
const context = defineModel<WritingContext>('context', {
  default: () => ({ ...EMPTY_CONTEXT }),
})

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
  selection.value = info
  if (!info) {
    menuPos.value = null
    return
  }
  menuPos.value = computeMenuPos(editor!, info)
}

/**
 * 用 ProseMirror 的 coordsAtPos 把菜单贴到选区上方；上方放不下就翻到下方。
 * coordsAtPos 返回的就是 viewport 坐标，配合 position: fixed 直接用，
 * 不用再减去任何祖先的 offset（菜单被 Teleport 到了 body 上）。
 */
function computeMenuPos(editor: Editor, info: SelectionInfo) {
  const start = editor.view.coordsAtPos(info.from)
  const end = editor.view.coordsAtPos(info.to)
  return placeMenu(start.top, start.left, end.right, end.bottom)
}

/** 工具栏按钮走这条：菜单贴着光标，而不是贴着整句。 */
function computeCursorMenuPos(editor: Editor, pos: number) {
  const c = editor.view.coordsAtPos(pos)
  return placeMenu(c.top, c.left, c.left, c.bottom)
}

function placeMenu(top: number, left: number, right: number, bottom: number) {
  // 菜单宽度按最宽的一档估：结果卡片比菜单列表宽
  const MENU_W = 320
  const MENU_H = 280
  const GAP = 8

  // 上方空间不够（顶到视口上沿）就翻到下方
  const placeBelow = top - MENU_H - GAP < 4
  const rawY = placeBelow ? bottom + GAP : top - MENU_H - GAP
  // 再夹进视口：面板展开后可能比 MENU_H 高，别让它整个跑到屏幕外
  const y = Math.max(4, Math.min(rawY, Math.max(4, window.innerHeight - MENU_H - 4)))

  // 水平方向夹进视口，别让菜单被裁掉
  const x = Math.max(8, Math.min((left + right) / 2 - MENU_W / 2, window.innerWidth - MENU_W - 8))

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
  menuPos.value = computeCursorMenuPos(editorInst, editorInst.state.selection.head)
  menuAction.value = action
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
  closeMenu()
})

// 暴露给父组件：翻译完成后整体替换英文段落
defineExpose({
  setHtml(html: string) {
    editor.value?.commands.setContent(html)
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
 * 工具栏上那排整句操作按钮。
 * 图标跟菜单项一致，但标签压到 2 个字——6 个按钮挤在半栏宽度里，
 * 用「改写语气」「检查语法」这种全称会把工具栏撑到换行。完整文案放在 title 上。
 */
const AI_BUTTONS: { action: AiAction; icon: string; label: string }[] = [
  { action: 'tone', icon: '🎨', label: '语气' },
  { action: 'improve', icon: '✏️', label: '润色' },
  { action: 'shorten', icon: '✂️', label: '精简' },
  { action: 'expand', icon: '➕', label: '扩写' },
  { action: 'grammar', icon: '🔍', label: '语法' },
  { action: 'instruction', icon: '💬', label: '指令' },
]

/** 完整文案，只进 title 提示。 */
const AI_TITLES: Record<AiAction, string> = {
  candidates: '替换选区',
  tone: '改写语气',
  improve: '润色',
  shorten: '精简',
  expand: '扩写',
  grammar: '检查语法',
  instruction: '自定义指令',
}

const editor = useEditor({
  content: '<p></p>',
  extensions: [
    StarterKit,
    createHighlightSentenceExtension(splitEnglishSentences, () => highlightRef.value),
  ],
  onUpdate({ editor }) {
    html.value = editor.getHTML()
    context.value = getWritingContext(editor, splitEnglishSentences)
    detectSelection(editor)
  },
  onSelectionUpdate({ editor }) {
    context.value = getWritingContext(editor, splitEnglishSentences)
    detectSelection(editor)
  },
  onCreate({ editor }) {
    html.value = editor.getHTML()
    context.value = getWritingContext(editor, splitEnglishSentences)
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

      <!-- 整句操作：作用于光标所在句，不需要选中 -->
      <div class="toolbar__group toolbar__group--ai">
        <button
          v-for="a in AI_BUTTONS"
          :key="a.action"
          type="button"
          :disabled="!cursorSentence"
          :title="`对光标所在句：${AI_TITLES[a.action]}`"
          @click="runOnCursorSentence(a.action)"
        >
          {{ a.icon }} {{ a.label }}
        </button>
      </div>
    </div>

    <EditorContent :editor="editor" />

    <!-- 选区上下文菜单：选中单词/短语/整句时自动弹出 -->
    <Teleport to="body">
      <div
        v-if="selection && menuPos"
        class="ai-menu-anchor"
        :style="{ top: `${menuPos.top}px`, left: `${menuPos.left}px` }"
      >
        <AiContextMenu
          :selection="selection"
          :chinese-sentence="chineseSentence"
          :auto-action="menuAction"
          @replace="applyReplacement"
          @close="closeMenu"
        />
      </div>
    </Teleport>
  </div>
</template>
