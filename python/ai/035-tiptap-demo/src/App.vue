<script setup lang="ts">
import { ref } from 'vue'
import TiptapEditor from './components/TiptapEditor.vue'
import { EMPTY_CONTEXT, type WritingContext } from './composables/useWritingContext'

const html = ref('')
const context = ref<WritingContext>({ ...EMPTY_CONTEXT })
</script>

<template>
  <div class="app">
    <header class="app__header">
      <h1>English Writing Pad</h1>
      <p class="app__hint">
        在左侧写作；右侧面板会随光标位置实时显示当前<strong>单词</strong>、<strong>句子</strong>和<strong>段落</strong>。
      </p>
    </header>

    <div class="app__main">
      <!-- 左栏：编辑器 -->
      <div class="app__editor">
        <TiptapEditor v-model:html="html" v-model:context="context" />
      </div>

      <!-- 右栏：上下文 + HTML 输出 -->
      <aside class="app__sidebar">
        <section class="ctx">
          <h2 class="ctx__title">当前段落</h2>
          <p v-if="context.paragraph" class="ctx__paragraph">
            <span>{{ context.paragraph.slice(0, context.paragraphHlStart) }}</span><mark
              class="ctx__hl ctx__hl--sentence"
              >{{ context.paragraph.slice(context.paragraphHlStart, context.paragraphHlEnd) }}</mark
            ><span>{{ context.paragraph.slice(context.paragraphHlEnd) }}</span>
          </p>
          <p v-else class="ctx__empty">光标不在任何段落中</p>
        </section>

        <section class="ctx">
          <h2 class="ctx__title">当前句子</h2>
          <p v-if="context.sentence" class="ctx__sentence">
            <span>{{ context.sentence.slice(0, context.sentenceHlStart) }}</span><mark
              class="ctx__hl ctx__hl--word"
              >{{ context.sentence.slice(context.sentenceHlStart, context.sentenceHlEnd) }}</mark
            ><span>{{ context.sentence.slice(context.sentenceHlEnd) }}</span>
          </p>
          <p v-else class="ctx__empty">光标不在任何句子中</p>
        </section>

        <section class="ctx">
          <h2 class="ctx__title">当前单词</h2>
          <p v-if="context.word" class="ctx__word">{{ context.word }}</p>
          <p v-else class="ctx__empty">—</p>
        </section>

        <section class="output">
          <div class="output__header">HTML（{{ html.length }} 字符）</div>
          <pre class="output__pre">{{ html }}</pre>
        </section>
      </aside>
    </div>
  </div>
</template>