<script setup lang="ts">
import { ref, watch } from 'vue'
import { EditorContent, useEditor } from '@tiptap/vue-3'
import StarterKit from '@tiptap/starter-kit'
import {
  EMPTY_CONTEXT,
  getWritingContext,
  splitChineseSentences,
  type WritingContext,
} from '../composables/useWritingContext'
import {
  createHighlightSentenceExtension,
  rerenderDecorations,
} from '../composables/highlightSentence'

const props = defineProps<{
  /** 来自英文侧的高亮：英文句子 i 时，此处把中文第 i 句高亮 */
  highlightedZhIndex: number
}>()

const html = defineModel<string>('html', { default: '' })
const context = defineModel<WritingContext>('context', {
  default: () => ({ ...EMPTY_CONTEXT }),
})

// 闭包给 ProseMirror 读
const highlightRef = ref(props.highlightedZhIndex)
watch(
  () => props.highlightedZhIndex,
  (v) => {
    highlightRef.value = v
    if (editor.value) rerenderDecorations(editor.value)
  },
)

const editor = useEditor({
  content: '<p></p>',
  extensions: [
    StarterKit.configure({
      heading: false,
      codeBlock: false,
      horizontalRule: false,
      blockquote: false,
      bulletList: false,
      orderedList: false,
    }),
    createHighlightSentenceExtension(splitChineseSentences, () => highlightRef.value),
  ],
  onUpdate({ editor }) {
    html.value = editor.getHTML()
    context.value = getWritingContext(editor, splitChineseSentences)
  },
  onSelectionUpdate({ editor }) {
    context.value = getWritingContext(editor, splitChineseSentences)
  },
  onCreate({ editor }) {
    html.value = editor.getHTML()
    context.value = getWritingContext(editor, splitChineseSentences)
  },
})
</script>

<template>
  <div class="editor-shell zh-shell">
    <div v-if="editor" class="toolbar">
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
      <div class="toolbar__group">
        <button type="button" :disabled="!editor.can().chain().focus().undo().run()"
          @click="editor.chain().focus().undo().run()" title="撤销 (Ctrl+Z)">↶</button>
        <button type="button" :disabled="!editor.can().chain().focus().redo().run()"
          @click="editor.chain().focus().redo().run()" title="重做 (Ctrl+Shift+Z)">↷</button>
      </div>
    </div>
    <EditorContent :editor="editor" />
  </div>
</template>