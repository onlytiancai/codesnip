<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import ChineseEditor from '../components/ChineseEditor.vue'
import TiptapEditor from '../components/TiptapEditor.vue'
import {
  EMPTY_CONTEXT,
  splitChineseSentences,
  splitEnglishSentences,
  type WritingContext,
} from '../composables/useWritingContext'
import { loadTones, useTones } from '../composables/useTones'
import { MODES, useMode } from '../composables/useMode'
import {
  translateSentences,
  TranslateHttpError,
  type AlignedSentence,
} from '../services/translate'
import { isToneId, type ToneId } from '../constants/tones'

/**
 * 编辑器页：左「源」右「译」。
 *
 * 两个方向的差别只有一处：哪边是源、哪边是译。
 *  - 中译英：源 = 中文，译 = 英文（英文侧带整套 AI 改写能力）
 *  - 英译中：源 = 英文，译 = 中文（英文侧只留「查词」）
 *
 * **两个编辑器的 DOM 位置是固定的**（中文栏、英文栏），
 * 靠 CSS `order` 把源栏排到左边 —— 这样切方向时编辑器不重建，内容不丢。
 */

// ===== 模式 =====
const { mode } = useMode()

const sourceLang = computed<'zh' | 'en'>(() => (mode.value === 'zh2en' ? 'zh' : 'en'))
const targetLang = computed<'zh' | 'en'>(() => (mode.value === 'zh2en' ? 'en' : 'zh'))

/** 源 / 译 各自的编辑器状态 */
const zhHtml = ref('')
const zhContext = ref<WritingContext>({ ...EMPTY_CONTEXT })
const enHtml = ref('')
const enContext = ref<WritingContext>({ ...EMPTY_CONTEXT })

const sourceHtml = computed(() => (sourceLang.value === 'zh' ? zhHtml.value : enHtml.value))
const sourceContext = computed(() =>
  sourceLang.value === 'zh' ? zhContext.value : enContext.value,
)
const targetContext = computed(() =>
  targetLang.value === 'zh' ? zhContext.value : enContext.value,
)

// ===== 对齐表：按方向各存一份（切方向不会串味）=====
const aligned = ref<Record<'zh2en' | 'en2zh', AlignedSentence[]>>({
  zh2en: [],
  en2zh: [],
})

const currentAligned = computed(() => aligned.value[mode.value])

/** 当前译文的第几句 —— 用来找它对应的源句 */
const targetSentenceIndex = computed(() => targetContext.value.sentenceIndex)

/** 当前英文句对应的中文原文（改写 API 的 sourceSentence；只在中译英下有意义） */
const currentChineseSentence = computed(() => {
  if (mode.value !== 'zh2en') return ''
  const i = targetSentenceIndex.value
  if (i < 0 || i >= currentAligned.value.length) return ''
  return currentAligned.value[i]?.source ?? ''
})

// ===== 翻译 =====
const translating = ref(false)
const progress = ref({ done: 0, total: 0 })
const errorMsg = ref('')

/** 翻译语气；空串 = 不指定，走后端默认 prompt */
const tone = ref<'' | ToneId>('')

/** 语气列表（label 从后端拉，各 provider 可改；拉不到用内置的） */
const { tones } = useTones()

onMounted(() => {
  void loadTones()
})

/** 两栏各自是源侧还是译侧（随方向变） */
const zhSide = computed<'source' | 'target'>(() =>
  sourceLang.value === 'zh' ? 'source' : 'target',
)
const enSide = computed<'source' | 'target'>(() =>
  sourceLang.value === 'en' ? 'source' : 'target',
)

// ===== 焦点跟踪：仅在另一侧获焦时才显示 cross-highlight =====
/** 焦点在源侧 → 高亮译侧；焦点在译侧 → 高亮源侧 */
const focusedSide = ref<'source' | 'target' | null>(null)

/** 中文栏该高亮哪一句（取决于中文是源还是译） */
const highlightedZhIndex = computed(() => {
  if (mode.value === 'zh2en') {
    // 中文是源 → 焦点在译侧时才高亮它
    return focusedSide.value === 'target' ? sourceContext.value.sentenceIndex : -1
  }
  // 中文是译 → 焦点在源侧时才高亮它
  return focusedSide.value === 'source' ? targetContext.value.sentenceIndex : -1
})

/** 英文栏该高亮哪一句 */
const highlightedEnIndex = computed(() => {
  if (mode.value === 'zh2en') {
    // 英文是译 → 焦点在源侧时才高亮它
    return focusedSide.value === 'source' ? sourceContext.value.sentenceIndex : -1
  }
  // 英文是源 → 焦点在译侧时才高亮它
  return focusedSide.value === 'target' ? targetContext.value.sentenceIndex : -1
})

// ===== 调试面板显隐 =====
const debugVisible = ref(false)

// ===== 调试日志：排查光标高亮错位 =====
watch(
  () => targetContext.value.sentenceIndex,
  (idx) => {
    if (currentAligned.value.length === 0) return
    console.groupCollapsed(`[debug] 译文光标 → 第 ${idx} 句`)
    console.log('aligned:', currentAligned.value.map((s) => s.source))
    console.log('current:', currentAligned.value[idx]?.target ?? '<out of range>')
    console.groupEnd()
  },
)

watch(
  () => sourceContext.value.sentenceIndex,
  (idx) => {
    if (currentAligned.value.length === 0) return
    console.groupCollapsed(`[debug] 源文光标 → 第 ${idx} 句`)
    console.log('aligned:', currentAligned.value.map((s) => s.source))
    console.log('current:', currentAligned.value[idx]?.source ?? '<out of range>')
    console.groupEnd()
  },
)

// 用于父组件把翻译结果写回译侧编辑器
const zhRef = ref<InstanceType<typeof ChineseEditor> | null>(null)
const enRef = ref<InstanceType<typeof TiptapEditor> | null>(null)

/** 译侧编辑器（中文栏 or 英文栏） */
const targetEditor = computed(() =>
  targetLang.value === 'zh' ? zhRef.value : enRef.value,
)

async function doTranslate() {
  errorMsg.value = ''

  // 1. 抓源侧全文，按句切
  const paragraphs = extractParagraphs(sourceHtml.value)
  if (paragraphs.length === 0) {
    errorMsg.value =
      sourceLang.value === 'zh'
        ? '中文编辑器为空，先写点东西再翻译。'
        : '英文编辑器为空，先写点东西再翻译。'
    return
  }

  const split = sourceLang.value === 'zh' ? splitChineseSentences : splitEnglishSentences
  const sentencesPerParagraph = paragraphs.map((p) => split(p).map((s) => s.text))

  const totalSentences = sentencesPerParagraph.reduce((n, arr) => n + arr.length, 0)
  if (totalSentences === 0) {
    errorMsg.value =
      sourceLang.value === 'zh'
        ? '未检测到中文句子（记得加句号 。！？.!?）。'
        : '未检测到英文句子（记得加句号 . ! ?）。'
    return
  }

  progress.value = { done: 0, total: totalSentences }
  translating.value = true

  // 2. 拍平所有句子，逐句并发翻译
  const flat: string[] = []
  sentencesPerParagraph.forEach((arr) => flat.push(...arr))

  try {
    const results = await translateSentences(
      flat,
      (done, total) => {
        progress.value = { done, total }
      },
      isToneId(tone.value) ? tone.value : undefined,
      sourceLang.value,
      targetLang.value,
    )
    aligned.value[mode.value] = results

    // 3. 拼成译文段落 HTML，写回译侧编辑器
    //    中文不用空格连接，英文用空格
    const sep = targetLang.value === 'zh' ? '' : ' '
    const targetParagraphsHtml = sentencesPerParagraph.map((arr, pIdx) => {
      let cursor = 0
      for (let p = 0; p < pIdx; p++) cursor += sentencesPerParagraph[p].length
      const segsForPara = results.slice(cursor, cursor + arr.length).map((s) => s.target)
      return `<p>${segsForPara.join(sep)}</p>`
    })
    targetEditor.value?.setHtml(targetParagraphsHtml.join(''))
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
 * 从编辑器的 HTML 里提取段落纯文本。
 * 用 DOMParser 解析，遍历 <p> 拿 textContent。
 */
function extractParagraphs(html: string): string[] {
  if (!html) return []
  const doc = new DOMParser().parseFromString(`<root>${html}</root>`, 'text/html')
  return Array.from(doc.querySelectorAll('p')).map((p) => p.textContent ?? '').filter(Boolean)
}

/**
 * 焦点离开本侧 column 时清掉本侧的 focus 标记。
 * 如果另一侧的 focusin 紧跟着触发，focusedSide 会被改成另一侧 → 这里不会再清。
 */
function onSideBlur(side: 'source' | 'target') {
  if (focusedSide.value === side) focusedSide.value = null
}
</script>

<template>
  <div class="app">
    <header class="app__header">
      <div class="app__header-row">
        <h1>{{ MODES[mode].title }}</h1>
        <div class="app__header-actions">
          <button
            type="button"
            class="btn-debug-toggle"
            :aria-pressed="debugVisible"
            @click="debugVisible = !debugVisible"
          >
            {{ debugVisible ? '隐藏调试' : '显示调试' }}
          </button>
        </div>
      </div>
    </header>

    <div class="app__main" :class="`app__main--${mode}`">
      <!-- 中文栏 -->
      <section
        class="app__col app__col--zh"
        @focusin="focusedSide = zhSide"
        @focusout="onSideBlur(zhSide)"
      >
        <div class="app__col-header">
          <span class="app__col-label">中文</span>
          <div v-if="sourceLang === 'zh'" class="app__col-actions">
            <select v-model="tone" class="tone-select" title="翻译语气">
              <option value="">默认语气</option>
              <option v-for="t in tones" :key="t.id" :value="t.id">{{ t.label }}</option>
            </select>
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
        </div>

        <ChineseEditor
          ref="zhRef"
          :highlighted-zh-index="highlightedZhIndex"
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

      <!-- 英文栏 -->
      <section
        class="app__col app__col--en"
        @focusin="focusedSide = enSide"
        @focusout="onSideBlur(enSide)"
      >
        <div class="app__col-header">
          <span class="app__col-label">英文</span>
          <div v-if="sourceLang === 'en'" class="app__col-actions">
            <select v-model="tone" class="tone-select" title="翻译语气">
              <option value="">默认语气</option>
              <option v-for="t in tones" :key="t.id" :value="t.id">{{ t.label }}</option>
            </select>
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
          <span v-if="debugVisible && currentAligned.length" class="app__col-meta">
            第 {{ targetSentenceIndex >= 0 ? targetSentenceIndex + 1 : '—' }} 句
            / 共 {{ currentAligned.length }} 句
          </span>
        </div>

        <TiptapEditor
          ref="enRef"
          :highlighted-en-index="highlightedEnIndex"
          :chinese-sentence="currentChineseSentence"
          :ai-enabled="mode === 'zh2en'"
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
