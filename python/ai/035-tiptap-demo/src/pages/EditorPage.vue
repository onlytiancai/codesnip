<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref, watch } from 'vue'
import ChineseEditor from '../components/ChineseEditor.vue'
import TiptapEditor from '../components/TiptapEditor.vue'
import {
  splitChineseSentences,
  splitEnglishSentences,
} from '../composables/useWritingContext'
import { loadTones, useTones } from '../composables/useTones'
import { useMode } from '../composables/useMode'
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
/** 中文侧光标所在句子的全局索引 */
const zhSentenceIndex = ref(-1)
const enHtml = ref('')
/** 英文侧光标所在句子的全局索引 */
const enSentenceIndex = ref(-1)

const sourceHtml = computed(() => (sourceLang.value === 'zh' ? zhHtml.value : enHtml.value))
const sourceSentenceIndex = computed(() =>
  sourceLang.value === 'zh' ? zhSentenceIndex.value : enSentenceIndex.value,
)
const targetSentenceIndex = computed(() =>
  targetLang.value === 'zh' ? zhSentenceIndex.value : enSentenceIndex.value,
)

// ===== 对齐表：按方向各存一份（切方向不会串味）=====
const aligned = ref<Record<'zh2en' | 'en2zh', AlignedSentence[]>>({
  zh2en: [],
  en2zh: [],
})

const currentAligned = computed(() => aligned.value[mode.value])

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

/**
 * 中文栏该高亮哪一句。
 *
 * 语义：**对侧获焦时，高亮「对侧光标对应的那一句」**。
 * 句子在两侧是一一对应的（对齐表就是按序号排的），所以直接用对侧的句子序号即可。
 */
const highlightedZhIndex = computed(() => {
  if (mode.value === 'zh2en') {
    // 英文是译 → 焦点在英文侧时，高亮英文光标对应的那句中文
    return focusedSide.value === 'target' ? targetSentenceIndex.value : -1
  }
  // 英文是源 → 焦点在英文侧时，高亮英文光标对应的那句中文
  return focusedSide.value === 'source' ? sourceSentenceIndex.value : -1
})

/**
 * 英文栏该高亮哪一句 —— 同上：高亮「对侧光标对应的那一句」。
 */
const highlightedEnIndex = computed(() => {
  if (mode.value === 'zh2en') {
    // 中文是源 → 焦点在中文侧时，高亮中文光标对应的那句英文
    return focusedSide.value === 'source' ? sourceSentenceIndex.value : -1
  }
  // 中文是译 → 焦点在中文侧时，高亮中文光标对应的那句英文
  return focusedSide.value === 'target' ? targetSentenceIndex.value : -1
})

// ===== 调试日志：排查光标高亮错位 =====
watch(
  () => targetSentenceIndex.value,
  (idx) => {
    if (currentAligned.value.length === 0) return
    console.groupCollapsed(`[debug] 译文光标 → 第 ${idx} 句`)
    console.log('aligned:', currentAligned.value.map((s) => s.source))
    console.log('current:', currentAligned.value[idx]?.target ?? '<out of range>')
    console.groupEnd()
  },
)

watch(
  () => sourceSentenceIndex.value,
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

    // 顺手把实时翻译的缓存也填上 —— 否则开实时翻译会把刚翻过的句子再翻一遍
    if (mode.value === 'zh2en') {
      liveCache.clear()
      for (const r of results) liveCache.set(liveCacheKey(r.source), r.target)
      lastLiveSentences = sentencesPerParagraph
    }

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

/** HTML 里还有没有文字（<p></p> 算空）。 */
function htmlHasText(html: string): boolean {
  return html.replace(/<[^>]*>/g, '').trim().length > 0
}

// ===== 实时翻译 =====

/** 停止输入多久后发翻译请求。 */
const LIVE_DEBOUNCE = 800

/** 开关存 localStorage —— 刷新后还保持上次的选择。 */
const LIVE_KEY = 'tiptap-demo.liveTranslate'

function readLiveFlag(): boolean {
  try {
    return localStorage.getItem(LIVE_KEY) === '1'
  } catch {
    return false
  }
}

const liveTranslate = ref(readLiveFlag())

function toggleLiveTranslate() {
  liveTranslate.value = !liveTranslate.value
  try {
    localStorage.setItem(LIVE_KEY, liveTranslate.value ? '1' : '0')
  } catch {
    // 存不了就算了，内存里的状态照样能用
  }
  if (liveTranslate.value) scheduleLiveTranslate()
}

/**
 * 源句 → 译文 的缓存。
 *
 * 同一句话不重复翻（打字过程中句子经常只改一两个字，没改过的句子直接复用）。
 * 语气算进 key —— 换了语气译文就变了。
 */
const liveCache = new Map<string, string>()

function liveCacheKey(sentence: string): string {
  return `${tone.value}\u0000${sentence}`
}

/** 上一次写回译侧的源段落句子结构 —— 用来算哪几段变了。 */
let lastLiveSentences: string[][] = []

/** 在飞的实时翻译 —— 新的一轮开始前先把旧的掐掉。 */
let liveAbort: AbortController | null = null
let liveTimer: ReturnType<typeof setTimeout> | undefined

onBeforeUnmount(() => {
  if (liveTimer) clearTimeout(liveTimer)
  liveAbort?.abort()
})

/** 中文改了 → 排一次实时翻译（防抖）。 */
function scheduleLiveTranslate() {
  if (!liveTranslate.value) return
  if (mode.value !== 'zh2en') return
  if (liveTimer) clearTimeout(liveTimer)
  liveTimer = setTimeout(() => {
    liveTimer = undefined
    void runLiveTranslate()
  }, LIVE_DEBOUNCE)
}

/**
 * 跑一轮实时翻译：只翻没翻过的句子，只改变了的段落。
 *
 * 写回是「按段落替换」而不是整篇 setContent —— 整篇重写会让译文框闪一下、
 * 滚动位置跳回顶部；只换改动的段落就没这些问题。
 */
async function runLiveTranslate() {
  if (!liveTranslate.value || mode.value !== 'zh2en') return
  if (translating.value) return // 手动翻译在跑，不插队

  const srcAtStart = zhHtml.value
  const paragraphs = extractParagraphs(srcAtStart)
  if (paragraphs.length === 0) return // 源侧空了 → 交给「源侧清空 → 译侧清空」那条规则

  const sentencesPerParagraph = paragraphs.map((p) =>
    splitChineseSentences(p).map((s) => s.text),
  )

  // 哪些句子还没翻过
  const missing: string[] = []
  for (const list of sentencesPerParagraph) {
    for (const s of list) {
      if (!liveCache.has(liveCacheKey(s)) && !missing.includes(s)) missing.push(s)
    }
  }

  if (missing.length > 0) {
    liveAbort?.abort()
    const ctl = new AbortController()
    liveAbort = ctl
    try {
      const results = await translateSentences(
        missing,
        undefined,
        isToneId(tone.value) ? tone.value : undefined,
        'zh',
        'en',
        ctl.signal,
      )
      for (const r of results) liveCache.set(liveCacheKey(r.source), r.target)
    } catch (err) {
      if (ctl.signal.aborted) return // 又改字了，等下一轮
      errorMsg.value =
        err instanceof TranslateHttpError
          ? `[${err.status}] ${err.message}`
          : err instanceof Error
            ? err.message
            : 'unknown error'
      return
    } finally {
      if (liveAbort === ctl) liveAbort = null
    }
  }

  // 翻译期间又改字了 → 这轮结果已经过期，交给下一轮
  if (zhHtml.value !== srcAtStart) return

  // 哪些段落变了
  const changed: number[] = []
  const structureChanged = sentencesPerParagraph.length !== lastLiveSentences.length
  for (let p = 0; p < sentencesPerParagraph.length; p++) {
    const before = lastLiveSentences[p] ?? []
    const after = sentencesPerParagraph[p]
    if (before.length !== after.length || before.some((s, i) => s !== after[i])) {
      changed.push(p)
    }
  }
  if (!structureChanged && changed.length === 0) return

  // 拼译文段落（英文用空格连接）
  const targetParagraphs = sentencesPerParagraph.map((list) =>
    list.map((s) => liveCache.get(liveCacheKey(s)) ?? '').join(' '),
  )

  if (structureChanged) {
    // 段落数变了 → 整体重建
    enRef.value?.setHtml(targetParagraphs.map((t) => `<p>${t}</p>`).join(''))
  } else {
    enRef.value?.applyParagraphs(targetParagraphs, changed)
  }
  lastLiveSentences = sentencesPerParagraph

  // 对齐表：跨栏高亮、「对应中文原文」都靠它
  aligned.value.zh2en = sentencesPerParagraph.flatMap((list) =>
    list.map((s) => ({ source: s, target: liveCache.get(liveCacheKey(s)) ?? '' })),
  )
}

// 中文改了 → 实时翻译（中译英方向）
watch(zhHtml, () => {
  if (!liveTranslate.value) return
  if (mode.value !== 'zh2en') return
  scheduleLiveTranslate()
})

// 换语气 → 译文全变，缓存作废重来
watch(tone, () => {
  if (!liveTranslate.value) return
  liveCache.clear()
  lastLiveSentences = []
  scheduleLiveTranslate()
})

// 换方向 → 实时翻译的缓存作废（中译英的译文对英译中没用）
watch(mode, () => {
  liveCache.clear()
  lastLiveSentences = []
})

// ===== 源侧清空 → 译侧跟着清空 =====
watch(sourceHtml, (html) => {
  if (htmlHasText(html)) return
  // 源侧已经空了，译文没有意义 → 一起清掉
  const targetHtml = targetLang.value === 'zh' ? zhHtml.value : enHtml.value
  if (!htmlHasText(targetHtml)) return
  targetEditor.value?.setHtml('')
  aligned.value[mode.value] = []
  // 实时翻译的「上次结构」也作废，否则重新写一句会被当成没变
  lastLiveSentences = []
})

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
        <h1>翻译工作台</h1>
        <div class="app__header-actions">
          <button
            type="button"
            class="btn-live-toggle"
            :aria-pressed="liveTranslate"
            :disabled="mode !== 'zh2en'"
            :title="
              mode !== 'zh2en'
                ? '实时翻译目前只支持中译英方向'
                : liveTranslate
                  ? '实时翻译已开启：中文改完自动翻英文'
                  : '实时翻译已关闭：需要点「翻译 →」'
            "
            @click="toggleLiveTranslate"
          >
            {{ liveTranslate ? '实时翻译 ●' : '实时翻译 ○' }}
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
          v-model:sentence-index="zhSentenceIndex"
        />

        <div v-if="errorMsg" class="app__error">{{ errorMsg }}</div>
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
        </div>

        <TiptapEditor
          ref="enRef"
          :highlighted-en-index="highlightedEnIndex"
          :chinese-sentence="currentChineseSentence"
          :ai-enabled="mode === 'zh2en'"
          v-model:html="enHtml"
          v-model:sentence-index="enSentenceIndex"
        />
      </section>
    </div>
  </div>
</template>
