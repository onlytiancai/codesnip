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
import { fetchCandidates, RewriteError } from '../services/rewrite'
import { matchCase } from '../utils/matchCase'

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

// ===== 单词选区检测 =====
const selectedWord = ref('')
const selectionRange = ref<{ from: number; to: number } | null>(null)
const paragraphText = ref('')

function detectSelection(editor: Editor) {
  if (!editor) return
  const { from, to } = editor.state.selection
  if (from === to) {
    selectedWord.value = ''
    selectionRange.value = null
    paragraphText.value = ''
    return
  }
  const text = editor.state.doc.textBetween(from, to, '', '')
  if (/^\w+$/.test(text)) {
    selectedWord.value = text
    selectionRange.value = { from, to }
    const ctxInfo = getWritingContext(editor, splitEnglishSentences)
    paragraphText.value = (ctxInfo.paragraph || '').slice(0, 200)
  } else {
    selectedWord.value = ''
    selectionRange.value = null
    paragraphText.value = ''
  }
}

// ===== popover 状态 =====
const popoverOpen = ref(false)
const candidates = ref<string[]>([])
const loading = ref(false)
const errorMsg = ref('')

async function openRewrite() {
  if (!selectedWord.value) return
  popoverOpen.value = true
  loading.value = true
  errorMsg.value = ''
  candidates.value = []
  try {
    candidates.value = await fetchCandidates({
      word: selectedWord.value,
      context: paragraphText.value || undefined,
      sourceSentence: props.chineseSentence || undefined,
    })
  } catch (err) {
    errorMsg.value =
      err instanceof RewriteError
        ? `[${err.status}] ${err.message}`
        : err instanceof Error
          ? err.message
          : 'unknown error'
  } finally {
    loading.value = false
  }
}

function closePopover() {
  popoverOpen.value = false
  candidates.value = []
  errorMsg.value = ''
}

function applyCandidate(candidate: string) {
  const editorInst = editor.value
  const range = selectionRange.value
  const original = selectedWord.value
  if (!editorInst || !range || !original) return
  const replacement = matchCase(original, candidate)
  editorInst
    .chain()
    .focus()
    .insertContentAt({ from: range.from, to: range.to }, replacement)
    .run()
  closePopover()
}

function onToolbarKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && popoverOpen.value) {
    closePopover()
  }
}

onBeforeUnmount(() => {
  closePopover()
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
  <div class="editor-shell en-shell" @keydown="onToolbarKeydown">
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

      <!-- AI：替换词 -->
      <div class="toolbar__group toolbar__group--ai">
        <button
          type="button"
          class="toolbar__btn-ai"
          :disabled="!selectedWord"
          :title="selectedWord ? `为「${selectedWord}」找替换词` : '先选中一个单词'"
          @click="popoverOpen ? closePopover() : openRewrite()"
        >
          {{ loading ? '… 生成中' : popoverOpen ? '× 关闭' : '✨ 替换词' }}
        </button>
      </div>
    </div>

    <EditorContent :editor="editor" />

    <!-- popover -->
    <div v-if="popoverOpen" class="rewrite-popover" role="dialog" aria-label="替换词候选">
      <div class="rewrite-popover__header">
        <span class="rewrite-popover__title">替换 <code>{{ selectedWord }}</code></span>
        <button type="button" class="rewrite-popover__close" aria-label="关闭" @click="closePopover">×</button>
      </div>

      <div v-if="loading" class="rewrite-popover__loading">模型生成中…</div>

      <div v-else-if="errorMsg" class="rewrite-popover__error">
        请求失败：{{ errorMsg }}
        <button type="button" class="rewrite-popover__retry" @click="openRewrite">重试</button>
      </div>

      <ul v-else class="rewrite-popover__list">
        <li v-for="c in candidates" :key="c">
          <button type="button" class="rewrite-popover__cand" @click="applyCandidate(c)">
            <span class="rewrite-popover__word">{{ matchCase(selectedWord, c) }}</span>
            <span class="rewrite-popover__hint">{{ c }}</span>
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>