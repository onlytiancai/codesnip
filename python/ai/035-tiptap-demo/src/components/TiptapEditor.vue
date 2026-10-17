<script setup lang="ts">
import { EditorContent, useEditor } from '@tiptap/vue-3'
import StarterKit from '@tiptap/starter-kit'
import { getWritingContext, EMPTY_CONTEXT, type WritingContext } from '../composables/useWritingContext'

const html = defineModel<string>('html', { default: '' })
const context = defineModel<WritingContext>('context', { default: () => ({ ...EMPTY_CONTEXT }) })

const editor = useEditor({
  content: `
    <p>Reading is the gateway to knowledge, allowing us to explore ideas and worlds far beyond our own experience. A single book can change the way we think, teaching us empathy, patience, and curiosity about the lives of others.</p>
    <p>When we read regularly, our minds become sharper and our vocabulary grows richer. Writers who read widely develop a natural sense of rhythm and tone that makes their own prose sing.</p>
  `,
  extensions: [StarterKit],
  // 文档每次更新都把最新 HTML 写回 v-model
  onUpdate({ editor }) {
    html.value = editor.getHTML()
    context.value = getWritingContext(editor)
  },
  // 光标变化时同步上下文
  onSelectionUpdate({ editor }) {
    context.value = getWritingContext(editor)
  },
  // 编辑器创建完成时也同步一次（确保初次渲染后 HTML 输出不是空）
  onCreate({ editor }) {
    html.value = editor.getHTML()
    context.value = getWritingContext(editor)
  },
})
</script>

<template>
  <div class="editor-shell">
    <div v-if="editor" class="toolbar">
      <!-- 文本格式 -->
      <div class="toolbar__group">
        <button
          type="button"
          :class="{ active: editor.isActive('bold') }"
          :disabled="!editor.can().chain().focus().toggleBold().run()"
          @click="editor.chain().focus().toggleBold().run()"
          title="加粗 (Ctrl+B)"
        >
          <strong>B</strong>
        </button>
        <button
          type="button"
          :class="{ active: editor.isActive('italic') }"
          :disabled="!editor.can().chain().focus().toggleItalic().run()"
          @click="editor.chain().focus().toggleItalic().run()"
          title="斜体 (Ctrl+I)"
        >
          <em>I</em>
        </button>
        <button
          type="button"
          :class="{ active: editor.isActive('strike') }"
          :disabled="!editor.can().chain().focus().toggleStrike().run()"
          @click="editor.chain().focus().toggleStrike().run()"
          title="删除线"
        >
          <s>S</s>
        </button>
        <button
          type="button"
          :class="{ active: editor.isActive('underline') }"
          :disabled="!editor.can().chain().focus().toggleUnderline().run()"
          @click="editor.chain().focus().toggleUnderline().run()"
          title="下划线 (Ctrl+U)"
        >
          <u>U</u>
        </button>
      </div>

      <!-- 标题 -->
      <div class="toolbar__group">
        <button
          type="button"
          :class="{ active: editor.isActive('heading', { level: 1 }) }"
          @click="editor.chain().focus().toggleHeading({ level: 1 }).run()"
          title="标题 1"
        >
          H1
        </button>
        <button
          type="button"
          :class="{ active: editor.isActive('heading', { level: 2 }) }"
          @click="editor.chain().focus().toggleHeading({ level: 2 }).run()"
          title="标题 2"
        >
          H2
        </button>
        <button
          type="button"
          :class="{ active: editor.isActive('heading', { level: 3 }) }"
          @click="editor.chain().focus().toggleHeading({ level: 3 }).run()"
          title="标题 3"
        >
          H3
        </button>
        <button
          type="button"
          :class="{ active: editor.isActive('paragraph') }"
          @click="editor.chain().focus().setParagraph().run()"
          title="正文"
        >
          P
        </button>
      </div>

      <!-- 列表 -->
      <div class="toolbar__group">
        <button
          type="button"
          :class="{ active: editor.isActive('bulletList') }"
          @click="editor.chain().focus().toggleBulletList().run()"
          title="无序列表"
        >
          • 列表
        </button>
        <button
          type="button"
          :class="{ active: editor.isActive('orderedList') }"
          @click="editor.chain().focus().toggleOrderedList().run()"
          title="有序列表"
        >
          1. 列表
        </button>
        <button
          type="button"
          :class="{ active: editor.isActive('blockquote') }"
          @click="editor.chain().focus().toggleBlockquote().run()"
          title="引用"
        >
          “ 引用
        </button>
      </div>

      <!-- 历史 -->
      <div class="toolbar__group">
        <button
          type="button"
          :disabled="!editor.can().chain().focus().undo().run()"
          @click="editor.chain().focus().undo().run()"
          title="撤销 (Ctrl+Z)"
        >
          ↶
        </button>
        <button
          type="button"
          :disabled="!editor.can().chain().focus().redo().run()"
          @click="editor.chain().focus().redo().run()"
          title="重做 (Ctrl+Shift+Z)"
        >
          ↷
        </button>
      </div>
    </div>

    <EditorContent :editor="editor" />
  </div>
</template>