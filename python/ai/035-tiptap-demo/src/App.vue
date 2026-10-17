<script setup lang="ts">
import { ref } from 'vue'
import TiptapEditor from './components/TiptapEditor.vue'

// 编辑器输出的实时 HTML
const html = ref('')
</script>

<template>
  <div class="app">
    <header class="app__header">
      <h1>Tiptap Demo</h1>
      <p class="app__hint">
        基于 <code>Vue 3 + @tiptap/vue-3 + @tiptap/starter-kit + Vite</code>。
        试试点击工具栏、输入文字、Ctrl+Z 撤销。
      </p>
    </header>

    <TiptapEditor v-model:html="html" />

    <section class="output">
      <div class="output__header">实时 HTML 输出（{{ html.length }} 字符）</div>
      <pre class="output__pre">{{ html }}</pre>
    </section>
  </div>
</template>