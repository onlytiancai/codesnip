<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue'
import { EditorContent, useEditor } from '@tiptap/vue-3'
import type { Editor } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import { getWritingContext, EMPTY_CONTEXT, type WritingContext } from '../composables/useWritingContext'
import { fetchCandidates, RewriteError } from '../services/rewrite'
import { matchCase } from '../utils/matchCase'

const html = defineModel<string>('html', { default: '' })
const context = defineModel<WritingContext>('context', { default: () => ({ ...EMPTY_CONTEXT }) })

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
    // 取当前段落文本作为 context（限长 200）
    const ctxInfo = getWritingContext(editor)
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

// ===== editor 实例 =====
const editor = useEditor({
  content: `
    <p>Reading is the gateway to knowledge, allowing us to explore ideas and worlds far beyond our own experience. A single book can change the way we think, teaching us empathy, patience, and curiosity about the lives of others.</p>
    <p>When we read regularly, our minds become sharper and our vocabulary grows richer. Writers who read widely develop a natural sense of rhythm and tone that makes their own prose sing.</p>
  `,
  extensions: [StarterKit],
  onUpdate({ editor }) {
    html.value = editor.getHTML()
    context.value = getWritingContext(editor)
    detectSelection(editor)
  },
  onSelectionUpdate({ editor }) {
    context.value = getWritingContext(editor)
    detectSelection(editor)
  },
  onCreate({ editor }) {
    html.value = editor.getHTML()
    context.value = getWritingContext(editor)
    detectSelection(editor)
  },
})
</script>

<template>
  <div class="editor-shell" @keydown="onToolbarKeydown">
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

      <!-- 标题 -->
      <div class="toolbar__group">
        <button type="button" :class="{ active: editor.isActive('heading', { level: 1 }) }"
          @click="editor.chain().focus().toggleHeading({ level: 1 }).run()" title="标题 1">H1</button>
        <button type="button" :class="{ active: editor.isActive('heading', { level: 2 }) }"
          @click="editor.chain().focus().toggleHeading({ level: 2 }).run()" title="标题 2">H2</button>
        <button type="button" :class="{ active: editor.isActive('heading', { level: 3 }) }"
          @click="editor.chain().focus().toggleHeading({ level: 3 }).run()" title="标题 3">H3</button>
        <button type="button" :class="{ active: editor.isActive('paragraph') }"
          @click="editor.chain().focus().setParagraph().run()" title="正文">P</button>
      </div>

      <!-- 列表 + 引用 -->
      <div class="toolbar__group">
        <button type="button" :class="{ active: editor.isActive('bulletList') }"
          @click="editor.chain().focus().toggleBulletList().run()" title="无序列表">• 列表</button>
        <button type="button" :class="{ active: editor.isActive('orderedList') }"
          @click="editor.chain().focus().toggleOrderedList().run()" title="有序列表">1. 列表</button>
        <button type="button" :class="{ active: editor.isActive('blockquote') }"
          @click="editor.chain().focus().toggleBlockquote().run()" title="引用">“ 引用</button>
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

    <!-- popover：从工具栏底部溢出显示 -->
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