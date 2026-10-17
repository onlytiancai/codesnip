<script setup lang="ts">
import { EditorContent, useEditor } from '@tiptap/vue-3'
import StarterKit from '@tiptap/starter-kit'

const html = defineModel<string>('html', { default: '' })

const editor = useEditor({
  content: `
    <h1>欢迎使用 Tiptap 👋</h1>
    <p>这是一个 <strong>最小可运行</strong> 的富文本编辑器 demo，基于 <code>Vue 3 + Tiptap + Vite</code>。</p>
    <p>试试下面这些操作：</p>
    <ul>
      <li>选中文字，使用工具栏加 <em>斜体</em> 或 <s>删除线</s></li>
      <li>切换 <strong>H1 / H2</strong> 标题</li>
      <li>用 <code>&lt;/&gt;</code> 按钮插入代码块</li>
    </ul>
    <blockquote>右侧会实时输出当前文档的 HTML。</blockquote>
  `,
  extensions: [StarterKit],
  // 文档每次更新都把最新 HTML 写回 v-model
  onUpdate({ editor }) {
    html.value = editor.getHTML()
  },
  // 编辑器创建完成时也同步一次（确保初次渲染后 HTML 输出不是空）
  onCreate({ editor }) {
    html.value = editor.getHTML()
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
          :class="{ active: editor.isActive('code') }"
          :disabled="!editor.can().chain().focus().toggleCode().run()"
          @click="editor.chain().focus().toggleCode().run()"
          title="行内代码"
        >
          &lt;/&gt;
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
      </div>

      <!-- 块级 -->
      <div class="toolbar__group">
        <button
          type="button"
          :class="{ active: editor.isActive('codeBlock') }"
          @click="editor.chain().focus().toggleCodeBlock().run()"
          title="代码块"
        >
          代码块
        </button>
        <button
          type="button"
          :class="{ active: editor.isActive('blockquote') }"
          @click="editor.chain().focus().toggleBlockquote().run()"
          title="引用"
        >
          引用
        </button>
        <button
          type="button"
          @click="editor.chain().focus().setHorizontalRule().run()"
          title="分隔线"
        >
          ―
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