<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import ChineseEditor from './components/ChineseEditor.vue'
import TiptapEditor from './components/TiptapEditor.vue'
import { EMPTY_CONTEXT, splitChineseSentences, type WritingContext } from './composables/useWritingContext'
import { translateSentences, TranslateHttpError, type AlignedSentence } from './services/translate'

// ===== 两边状态 =====
const zhHtml = ref('')
const zhContext = ref<WritingContext>({ ...EMPTY_CONTEXT })
const enHtml = ref('')
const enContext = ref<WritingContext>({ ...EMPTY_CONTEXT })

// ===== 联动索引：左 → 右 / 右 → 左 =====
const zhSentenceIndex = computed(() => zhContext.value.sentenceIndex)
const enSentenceIndex = computed(() => enContext.value.sentenceIndex)

// 当前英文句子的中文原文（用于改写 API 的 sourceSentence）
const currentChineseSentence = computed(() => {
  const i = enSentenceIndex.value
  if (i < 0 || i >= alignedSentences.value.length) return ''
  return alignedSentences.value[i]?.zh ?? ''
})

// ===== 翻译 =====
const alignedSentences = ref<AlignedSentence[]>([])
const translating = ref(false)
const progress = ref({ done: 0, total: 0 })
const errorMsg = ref('')

// ===== 焦点跟踪：仅在另一侧获焦时才显示 cross-highlight =====
// 默认 / 翻译完成后两边都没获焦 → 两边都不高亮
// 焦点在中文 → 英文侧高亮（zhSentenceIndex）
// 焦点在英文 → 中文侧高亮（enSentenceIndex）
const focusedSide = ref<'zh' | 'en' | null>(null)

const displayedZhHighlight = computed(() =>
  focusedSide.value === 'en' ? enSentenceIndex.value : -1,
)
const displayedEnHighlight = computed(() =>
  focusedSide.value === 'zh' ? zhSentenceIndex.value : -1,
)

// ===== 调试面板显隐 =====
const debugVisible = ref(false)

// ===== 调试日志：排查英文光标高亮错位 =====
watch(enSentenceIndex, (idx) => {
  // 仅在英文端有 mapped 数据时打印（翻译前都是空）
  if (alignedSentences.value.length === 0) return
  console.groupCollapsed(`[debug] English cursor → sentence #${idx}`)
  console.log('enSentenceIndex (current):', idx)
  console.log('zhSentenceIndex (cross-highlight source):', zhSentenceIndex.value)
  console.log('current EN sentence:', alignedSentences.value[idx]?.en ?? '<out of range>')
  console.log('aligned ZH array:', alignedSentences.value.map((s) => s.zh))
  console.log('aligned EN array:', alignedSentences.value.map((s) => s.en))
  console.log('mapped ZH for current idx:', alignedSentences.value[idx]?.zh ?? '<out of range>')
  console.groupEnd()
})

watch(zhSentenceIndex, (idx) => {
  if (alignedSentences.value.length === 0) return
  console.groupCollapsed(`[debug] Chinese cursor → sentence #${idx}`)
  console.log('zhSentenceIndex (current):', idx)
  console.log('enSentenceIndex (cross-highlight source):', enSentenceIndex.value)
  console.log('current ZH sentence:', alignedSentences.value[idx]?.zh ?? '<out of range>')
  console.log('mapped EN for current idx:', alignedSentences.value[idx]?.en ?? '<out of range>')
  console.groupEnd()
})

watch(alignedSentences, (sents) => {
  console.log('[debug] alignedSentences updated →', sents.length, 'sentences')
  console.table(sents)
}, { deep: true })

// 用于父组件把翻译结果写回英文编辑器
const tiptapRef = ref<InstanceType<typeof TiptapEditor> | null>(null)
const chineseRef = ref<InstanceType<typeof ChineseEditor> | null>(null)

async function doTranslate() {
  errorMsg.value = ''

  // 1. 抓中文段落全文，按句切
  //    zhContext.paragraph 是当前光标所在段落；这里要遍历所有段落
  const zhEditor = chineseRef.value
  if (!zhEditor) return

  // 借 editor 拿到完整 doc 的 textContent
  // ChineseEditor 暴露 editor 实例 → 通过 getJSON 拿到 paragraphs
  // 简单做法：让 ChineseEditor 暴露一个 fullText 方法
  // 这里通过 zhHtml 解析更直接（HTML → 纯文本）
  const allParagraphs = extractParagraphsFromZh()
  if (allParagraphs.length === 0) {
    errorMsg.value = '中文编辑器为空，先写点东西再翻译。'
    return
  }

  const sentencesPerParagraph = allParagraphs.map((p) => splitChineseSentences(p).map((s) => s.text))

  const totalSentences = sentencesPerParagraph.reduce((n, arr) => n + arr.length, 0)
  if (totalSentences === 0) {
    errorMsg.value = '未检测到中文句子（记得加句号 。！？.!?）。'
    return
  }

  progress.value = { done: 0, total: totalSentences }
  translating.value = true

  // 2. 拍平所有句子，逐句并发翻译
  const flat: string[] = []
  sentencesPerParagraph.forEach((arr) => flat.push(...arr))

  try {
    const results = await translateSentences(flat, (done, total) => {
      progress.value = { done, total }
    })
    alignedSentences.value = results

    // 3. 拼成英文段落 HTML，写回 TiptapEditor
    const enParagraphsHtml = sentencesPerParagraph.map((arr, pIdx) => {
      let cursor = 0
      for (let p = 0; p < pIdx; p++) cursor += sentencesPerParagraph[p].length
      const segsForPara = results.slice(cursor, cursor + arr.length).map((s) => s.en)
      return `<p>${segsForPara.join(' ')}</p>`
    })
    const html = enParagraphsHtml.join('')
    tiptapRef.value?.setHtml(html)
  } catch (err) {
    errorMsg.value =
      err instanceof TranslateHttpError
        ? `[${err.status}] ${err.message}`
        : err instanceof Error
          ? err.message
          : 'unknown error'
  } finally {
    translating.value = false
  }
}

/**
 * 从 ChineseEditor 的 HTML 中提取段落纯文本。
 * 简单做法：用 DOMParser 解析，遍历 <p> 拿 textContent。
 */
function extractParagraphsFromZh(): string[] {
  const html = zhHtml.value
  if (!html) return []
  const doc = new DOMParser().parseFromString(`<root>${html}</root>`, 'text/html')
  return Array.from(doc.querySelectorAll('p')).map((p) => p.textContent ?? '').filter(Boolean)
}

/**
 * 焦点离开本侧 column 时清掉本侧的 focus 标记。
 * 如果另一侧的 focusin 紧跟着触发，focusedSide 会被改成另一侧 → 这里不会再清。
 */
function onSideBlur(side: 'zh' | 'en') {
  if (focusedSide.value === side) focusedSide.value = null
}
</script>

<template>
  <div class="app">
    <header class="app__header">
      <div class="app__header-row">
        <h1>中英翻译写作台</h1>
        <button
          type="button"
          class="btn-debug-toggle"
          :aria-pressed="debugVisible"
          @click="debugVisible = !debugVisible"
        >
          {{ debugVisible ? '隐藏调试' : '显示调试' }}
        </button>
      </div>
    </header>

    <div class="app__main">
      <!-- 左：中文 -->
      <section
        class="app__col app__col--zh"
        @focusin="focusedSide = 'zh'"
        @focusout="onSideBlur('zh')"
      >
        <div class="app__col-header">
          <span class="app__col-label">中文</span>
          <button
            type="button"
            class="btn-translate"
            :disabled="translating"
            @click="doTranslate"
          >
            {{
              translating
                ? `翻译中 ${progress.done}/${progress.total}`
                : '翻译 →'
            }}
          </button>
        </div>

        <ChineseEditor
          ref="chineseRef"
          :highlighted-zh-index="displayedZhHighlight"
          v-model:html="zhHtml"
          v-model:context="zhContext"
        />

        <div v-if="errorMsg" class="app__error">{{ errorMsg }}</div>

        <section v-if="debugVisible" class="ctx">
          <h2 class="ctx__title">当前中文句子</h2>
          <p v-if="zhContext.sentence" class="ctx__zh">
            {{ zhContext.sentence }}
          </p>
          <p v-else class="ctx__empty">光标不在任何句子中</p>
        </section>
      </section>

      <!-- 右：英文 -->
      <section
        class="app__col app__col--en"
        @focusin="focusedSide = 'en'"
        @focusout="onSideBlur('en')"
      >
        <div class="app__col-header">
          <span class="app__col-label">英文</span>
          <span v-if="debugVisible && alignedSentences.length" class="app__col-meta">
            第 {{ enSentenceIndex >= 0 ? enSentenceIndex + 1 : '—' }} 句
            / 共 {{ alignedSentences.length }} 句
          </span>
        </div>

        <TiptapEditor
          ref="tiptapRef"
          :highlighted-en-index="displayedEnHighlight"
          :chinese-sentence="currentChineseSentence"
          v-model:html="enHtml"
          v-model:context="enContext"
        />

        <section v-if="debugVisible" class="ctx">
          <h2 class="ctx__title">当前段落 / 句子 / 单词</h2>
          <p v-if="enContext.paragraph" class="ctx__paragraph">
            <span>{{ enContext.paragraph.slice(0, enContext.paragraphHlStart) }}</span
            ><mark class="ctx__hl ctx__hl--sentence">{{
              enContext.paragraph.slice(enContext.paragraphHlStart, enContext.paragraphHlEnd)
            }}</mark
            ><span>{{ enContext.paragraph.slice(enContext.paragraphHlEnd) }}</span>
          </p>
          <p v-else class="ctx__empty">光标不在任何段落中</p>

          <p v-if="enContext.sentence" class="ctx__sentence">
            <span>{{ enContext.sentence.slice(0, enContext.sentenceHlStart) }}</span
            ><mark class="ctx__hl ctx__hl--word">{{
              enContext.sentence.slice(enContext.sentenceHlStart, enContext.sentenceHlEnd)
            }}</mark
            ><span>{{ enContext.sentence.slice(enContext.sentenceHlEnd) }}</span>
          </p>

          <p v-if="enContext.word" class="ctx__word">{{ enContext.word }}</p>
        </section>

        <section v-if="debugVisible && currentChineseSentence" class="ctx">
          <h2 class="ctx__title">对应中文原文（传给改写 API）</h2>
          <p class="ctx__zh">{{ currentChineseSentence }}</p>
        </section>
      </section>
    </div>
  </div>
</template>