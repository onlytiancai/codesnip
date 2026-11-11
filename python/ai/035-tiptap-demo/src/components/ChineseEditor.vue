<script setup lang="ts">
import { ref, watch } from 'vue'
import { EditorContent, useEditor } from '@tiptap/vue-3'
import StarterKit from '@tiptap/starter-kit'
import {
  getSentenceIndex,
  splitChineseSentences,
} from '../composables/useWritingContext'
import {
  createContextHighlightExtension,
  rerenderDecorations,
} from '../composables/highlightContext'
import { useCopyContent } from '../composables/useCopyContent'

const props = withDefaults(
  defineProps<{
    /** 来自英文侧的高亮：英文句子 i 时，此处把中文第 i 句高亮 */
    highlightedZhIndex: number
    /**
     * 是不是源侧。源侧给「清空」，译侧给「复制」——
     * 切方向时两个框的按钮跟着互换。
     */
    isSource?: boolean
  }>(),
  { isSource: true },
)

const html = defineModel<string>('html', { default: '' })
/** 光标所在句子的文档级全局索引（跨面板对齐的唯一依据） */
const sentenceIndex = defineModel<number>('sentenceIndex', { default: -1 })

// 闭包给 ProseMirror 读
const highlightRef = ref(props.highlightedZhIndex)
watch(
  () => props.highlightedZhIndex,
  (v) => {
    highlightRef.value = v
    if (editor.value) rerenderDecorations(editor.value)
  },
)

// 父组件把译文整体写进来（英译中模式下中文栏是译侧）
defineExpose({
  setHtml(html: string) {
    editor.value?.commands.setContent(html)
  },
})

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
    createContextHighlightExtension({
      splitFn: splitChineseSentences,
      getCrossIndex: () => highlightRef.value,
      // 中文框只要紫色的跨栏高亮，不要段落/句子/单词三层
      showContextLayers: () => false,
    }),
  ],
  onUpdate({ editor }) {
    html.value = editor.getHTML()
    sentenceIndex.value = getSentenceIndex(editor.state, splitChineseSentences)
  },
  onSelectionUpdate({ editor }) {
    sentenceIndex.value = getSentenceIndex(editor.state, splitChineseSentences)
  },
  onCreate({ editor }) {
    html.value = editor.getHTML()
    sentenceIndex.value = getSentenceIndex(editor.state, splitChineseSentences)
  },
})

// ===== 复制内容 =====
const { copied, copyContent } = useCopyContent(() => editor.value)
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
      <div class="toolbar__group toolbar__group--end">
        <!-- 源侧：清空；译侧：复制 -->
        <button
          v-if="isSource"
          type="button"
          class="toolbar__btn-clear"
          title="清空中文内容"
          :disabled="!editor.getText().trim()"
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
          :title="copied ? '已复制到剪贴板' : '复制中文内容'"
          @click="copyContent"
        >
          {{ copied ? '✓ 已复制' : '复制' }}
        </button>
      </div>
    </div>
    <EditorContent :editor="editor" />
  </div>
</template>