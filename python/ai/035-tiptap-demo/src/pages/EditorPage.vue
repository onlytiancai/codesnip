<script setup lang="ts">
import { computed, nextTick, onMounted, onBeforeUnmount, ref, watch } from 'vue'
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
  translateSentencesStream,
  TranslateHttpError,
  type AlignedSentence,
} from '../services/translate'
import { isToneId, type ToneId } from '../constants/tones'
import { escapeHtml } from '../utils/html'

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

/**
 * 每个方向各存一份两栏内容。
 *
 * 中译英和英译中是**两套互不相干的状态**：在中译英的中文栏输入的字，
 * 切到英译中不会带过去（英译中的中文栏是空的），英文栏同理。
 * 每个方向的内容各自存着，切回来还在。
 */
const modeHtml = ref<Record<'zh2en' | 'en2zh', { zh: string; en: string }>>({
  zh2en: { zh: '', en: '' },
  en2zh: { zh: '', en: '' },
})

/**
 * 正在换方向（要把旧方向的内容存起来、把新方向的内容装进去）。
 *
 * 这期间两栏的 html 会整体突变 —— 别让「源侧清空 → 译侧跟着清」
 * 「实时翻译」这些联动规则误触发。
 */
let restoring = false

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

/**
 * 每个英文句子对应的中文（按全局句子索引）—— 英文框的悬停气泡用。
 *
 * 中译英：中文是**源句** → aligned[i].source
 * 英译中：中文是**译文** → aligned[i].target
 */
const chineseSentences = computed(() =>
  mode.value === 'zh2en'
    ? currentAligned.value.map((a) => a.source)
    : currentAligned.value.map((a) => a.target),
)

// ===== 翻译 =====
const translating = ref(false)
const progress = ref({ done: 0, total: 0 })
const errorMsg = ref('')

/**
 * 正在翻译的句子（**源侧**的全局句子索引）。
 *
 * 源侧每翻一句就把那句标成 loading；翻完就撤掉。
 * 索引用的是编辑器里那套「跨段落累计」的编号 —— 和高亮、跨栏对齐是同一套。
 */
const loadingIndexes = ref<number[]>([])
/** 翻译失败的句子（源侧的全局句子索引）—— 标红 */
const failedIndexes = ref<number[]>([])

/**
 * 谁在占用 loadingIndexes。
 *
 * 手动翻译和实时翻译可能前后脚接力（手动翻译会掐掉实时翻译）——
 * 没有这个标记的话，实时翻译收尾时会把手动翻译刚挂上的 loading 一起清掉。
 */
let loadingOwner: 'manual' | 'live' | null = null

/** 本轮翻译的取消开关 —— 切方向 / 卸载时把在飞的请求掐掉 */
let translateAbort: AbortController | null = null

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

/**
 * 手动翻译：一句一句翻，译完一句就显示一句。
 *
 * 流程：
 *   1. 源侧切句 → 摊平成一个数组（数组下标 = 源侧的全局句子索引）
 *   2. 译侧先铺一层占位（每句一个「…」），段落结构和源侧一一对应
 *   3. 并发 3 逐句翻；每翻完一句就把译侧那一段重写一次（占位换成译文）
 *   4. 源侧正在翻的那几句挂 loading，失败的标红
 *
 * 译侧**不整体重写** —— 整体 setContent 会让译文框闪一下、滚动跳回顶部；
 * 只替换变的那一段就没事。
 */
async function doTranslate() {
  errorMsg.value = ''
  if (translating.value) return

  // 记下发起时的方向 —— 翻译期间切了方向，这次结果就不写进任何一栏
  const modeAtStart = mode.value
  const srcLang = sourceLang.value
  const tgtLang = targetLang.value

  // 1. 抓源侧全文，按句切
  const paragraphs = extractParagraphs(sourceHtml.value)
  if (paragraphs.length === 0) {
    errorMsg.value =
      srcLang === 'zh'
        ? '中文编辑器为空，先写点东西再翻译。'
        : '英文编辑器为空，先写点东西再翻译。'
    return
  }

  const split = srcLang === 'zh' ? splitChineseSentences : splitEnglishSentences
  const sentencesPerParagraph = paragraphs.map((p) => split(p).map((s) => s.text))

  // 摊平：flat[i] 是第 i 句，paraOf[i] 是它属于第几段
  const flat: string[] = []
  const paraOf: number[] = []
  sentencesPerParagraph.forEach((arr, pi) => {
    for (const s of arr) {
      flat.push(s)
      paraOf.push(pi)
    }
  })

  const total = flat.length
  if (total === 0) {
    errorMsg.value =
      srcLang === 'zh'
        ? '未检测到中文句子（记得加句号 。！？.!?）。'
        : '未检测到英文句子（记得加句号 . ! ?）。'
    return
  }

  progress.value = { done: 0, total }
  translating.value = true
  loadingIndexes.value = []
  loadingOwner = 'manual'
  failedIndexes.value = []
  // 记下译侧原来的内容 —— 整体失败（比如配额用尽）时好还原回去
  // ⚠️ 得在清空对齐表**之前**存
  const prevTargetHtml = targetLang.value === 'zh' ? zhHtml.value : enHtml.value
  const prevAligned = aligned.value[modeAtStart]

  // 翻译期间对齐表是空的 —— 免得跨栏高亮指着上一轮的句子
  aligned.value[modeAtStart] = []

  // 手动翻译和实时翻译别同时往译侧写 —— 把在飞的实时翻译掐掉
  liveAbort?.abort()

  // 2. 译侧先铺占位
  const sep = tgtLang === 'zh' ? '' : ' '
  const PLACEHOLDER = '…'
  const FAILED_MARK = '⚠'
  const done = new Array<boolean>(total).fill(false)
  const failed = new Array<boolean>(total).fill(false)
  const targets = new Array<string>(total).fill('')

  /** 第 pi 段现在的样子（已翻的用译文，没翻的用占位，失败的用 ⚠） */
  const buildPara = (pi: number): string => {
    const out: string[] = []
    sentencesPerParagraph[pi].forEach((_, i) => {
      const gi = flatIndexOf(pi, i)
      out.push(failed[gi] ? FAILED_MARK : done[gi] ? targets[gi] : PLACEHOLDER)
    })
    return out.join(sep)
  }

  /** 第 pi 段第 i 句在 flat 里的下标 */
  function flatIndexOf(pi: number, i: number): number {
    let n = 0
    for (let p = 0; p < pi; p++) n += sentencesPerParagraph[p].length
    return n + i
  }

  const paraTexts = sentencesPerParagraph.map((_, pi) => buildPara(pi))

  // 3. 并发翻，一句一回调
  const ctl = new AbortController()
  translateAbort = ctl
  /** 单句失败时的错误信息（登录过期 / 配额用尽这类会整体失败） */
  let fatalMsg = ''
  /** 整体失败（登录过期 / 配额用尽）→ 译侧还原回原来的内容 */
  let fatal = false

  try {
    // 译侧先铺占位（放在 try 里 —— 万一炸了，finally 照样会解锁编辑器）
    targetEditor.value?.setHtml(paraTexts.map((t) => `<p>${escapeHtml(t)}</p>`).join(''))

    await translateSentencesStream(
      flat,
      // 英译中没有语气（那边就是普通翻译）—— 不传 tone 走后端默认 prompt
      srcLang === 'zh' && isToneId(tone.value) ? tone.value : undefined,
      srcLang,
      tgtLang,
      ctl.signal,
      {
        onStart: (i) => {
          loadingIndexes.value = [...loadingIndexes.value, i]
        },
        onSettled: (i, r, err) => {
          // 中途切了方向 → 这轮结果整个作废
          if (ctl.signal.aborted) return

          loadingIndexes.value = loadingIndexes.value.filter((x) => x !== i)
          progress.value = { done: progress.value.done + 1, total }

          if (r) {
            done[i] = true
            targets[i] = r.target
          } else {
            failed[i] = true
            failedIndexes.value = [...failedIndexes.value, i]
            if (!fatalMsg) {
              fatalMsg =
                err instanceof TranslateHttpError
                  ? `[${err.status}] ${err.message}`
                  : err instanceof Error
                    ? err.message
                    : 'unknown error'
              // 登录过期 / 配额用尽 —— 再翻剩下的也是白翻，掐掉
              if (err instanceof TranslateHttpError && (err.status === 401 || err.status === 403)) {
                fatal = true
                ctl.abort()
              }
            }
          }

          // 这一句所在的那一段重写一次（只改这一段，译文框不闪）
          const pi = paraOf[i]
          paraTexts[pi] = buildPara(pi)
          targetEditor.value?.applyParagraphs(paraTexts, [pi])
        },
      },
    )

    // 翻译期间切了方向 → 结果作废，不写进任何一栏
    if (mode.value !== modeAtStart) return

    // 整体失败（登录过期 / 配额用尽）→ 译侧还原回原来的内容，别留一堆占位符
    if (fatal) {
      targetEditor.value?.setHtml(prevTargetHtml || '<p></p>')
      aligned.value[modeAtStart] = prevAligned
      errorMsg.value = fatalMsg
      return
    }

    // 对齐表：跨栏高亮、「对应中文原文」都靠它
    aligned.value[modeAtStart] = flat.map((s, i) => ({ source: s, target: targets[i] }))

    // 顺手把实时翻译的缓存也填上 —— 否则开实时翻译会把刚翻过的句子再翻一遍
    if (mode.value === 'zh2en') {
      liveCache.clear()
      for (let i = 0; i < flat.length; i++) {
        if (done[i]) liveCache.set(liveCacheKey(flat[i]), targets[i])
      }
      lastLiveSentences = sentencesPerParagraph
    }

    if (fatalMsg) errorMsg.value = fatalMsg
  } catch (err) {
    errorMsg.value =
      err instanceof TranslateHttpError
        ? `[${err.status}] ${err.message}`
        : err instanceof Error
          ? err.message
          : 'unknown error'
  } finally {
    translating.value = false
    if (loadingOwner === 'manual') {
      loadingIndexes.value = []
      loadingOwner = null
    }
    translateAbort = null
  }
}

/**
 * 从编辑器的 HTML 里提取段落纯文本。
 * 用 DOMParser 解析，遍历 <p> 和标题拿 textContent。
 *
 * 标题也算 —— 英文框允许标题（StarterKit 全量），
 * 漏掉的话标题里的句子压根不会被翻译。
 */
function extractParagraphs(html: string): string[] {
  if (!html) return []
  const doc = new DOMParser().parseFromString(`<root>${html}</root>`, 'text/html')
  const paras = Array.from(doc.querySelectorAll('p, h1, h2, h3, h4, h5, h6'))
    .map((p) => p.textContent ?? '')
    .filter(Boolean)
  if (paras.length) return paras
  // 兜底：一个 <p> 都没有 —— 内容全在代码块 / 引用 / 列表里。
  // （粘贴已经去格式了，这里是防手敲 ``` 或历史内容）
  // 按顶层块取文字，至少别报「编辑器为空」。
  const root = doc.body.firstElementChild
  if (!root) return []
  return Array.from(root.children)
    .map((el) => el.textContent ?? '')
    .filter(Boolean)
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
  translateAbort?.abort()
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

  // 哪些句子还没翻过（顺便记下它们在全文里的下标 —— 那是源侧的全局句子索引）
  const flatSentences: string[] = []
  for (const list of sentencesPerParagraph) flatSentences.push(...list)

  const missing: string[] = []
  const missingIdx: number[] = []
  for (let i = 0; i < flatSentences.length; i++) {
    const s = flatSentences[i]
    if (!liveCache.has(liveCacheKey(s)) && !missing.includes(s)) {
      missing.push(s)
      missingIdx.push(i)
    }
  }

  if (missing.length > 0) {
    liveAbort?.abort()
    const ctl = new AbortController()
    liveAbort = ctl
    // 正在翻的那几句，源侧挂 loading
    loadingIndexes.value = missingIdx
    loadingOwner = 'live'
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
      // 只有还是自己在占用 loadingIndexes 时才清 —— 手动翻译可能已经接手了
      if (loadingOwner === 'live') {
        loadingIndexes.value = []
        loadingOwner = null
      }
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
  // 换方向时两栏内容会整体替换 —— 那不是用户编辑，别触发实时翻译
  if (restoring) return
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

/**
 * 换方向 = 换一套**独立**的编辑器状态。
 *
 * 中译英和英译中互不继承：在中译英的中文栏输入的字，切到英译中不会带过去
 * （英译中的中文栏是空的），英文栏同理。每个方向的内容各自存着，切回来还在。
 */
watch(mode, (next, prev) => {
  // 1. 当前两栏内容存回上一个方向
  modeHtml.value[prev] = { zh: zhHtml.value, en: enHtml.value }

  // 2. 在飞的实时翻译要掐掉 —— 它会把译文写进英文栏，那栏马上要换主人了
  liveAbort?.abort()

  // 手动翻译也一样：这轮结果属于上一个方向，别写进新方向的栏
  translateAbort?.abort()

  // 3. 装上新方向的内容（没存过就是空的）
  restoring = true
  const nextState = modeHtml.value[next]
  zhHtml.value = nextState.zh
  enHtml.value = nextState.en
  zhRef.value?.setHtml(nextState.zh || '<p></p>')
  enRef.value?.setHtml(nextState.en || '<p></p>')

  // 4. 光标 / 焦点 / 报错状态重置
  zhSentenceIndex.value = -1
  enSentenceIndex.value = -1
  focusedSide.value = null
  errorMsg.value = ''
  progress.value = { done: 0, total: 0 }

  // 5. 实时翻译的缓存作废（中译英的译文对英译中没用）
  liveCache.clear()
  lastLiveSentences = []

  // 6. 装完再放开联动规则 —— 两栏 html 突变触发的 watcher 跟这一轮 flush 一起跑
  void nextTick(() => {
    restoring = false
  })
})

// ===== 源侧清空 → 译侧跟着清空 =====
/**
 * 源侧被清空时，译文没有意义 → 一起清掉。
 *
 * 分别监听两栏（而不是监听 computed 的 sourceHtml）：
 * sourceHtml 随 mode 变，**切方向**也会触发 —— 新方向的源侧往往是空的，
 * 那样会在切方向的瞬间把译文（也就是另一栏的用户内容）误清掉。
 */
function onSourceCleared(html: string) {
  if (htmlHasText(html)) return
  const targetHtml = targetLang.value === 'zh' ? zhHtml.value : enHtml.value
  if (!htmlHasText(targetHtml)) return
  targetEditor.value?.setHtml('')
  aligned.value[mode.value] = []
  // 实时翻译的「上次结构」也作废，否则重新写一句会被当成没变
  lastLiveSentences = []
}

watch(zhHtml, (html) => {
  if (restoring) return
  if (sourceLang.value === 'zh') onSourceCleared(html)
})
watch(enHtml, (html) => {
  if (restoring) return
  if (sourceLang.value === 'en') onSourceCleared(html)
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
            <select v-model="tone" class="tone-select" title="翻译语气" :disabled="translating">
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
          :is-source="zhSide === 'source'"
          :loading-indexes="zhSide === 'source' ? loadingIndexes : []"
          :failed-indexes="zhSide === 'source' ? failedIndexes : []"
          :editable="!translating"
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
            <!-- 英译中不选语气 —— 就是普通翻译 -->
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
          :is-source="enSide === 'source'"
          :minimal="mode === 'en2zh'"
          :loading-indexes="enSide === 'source' ? loadingIndexes : []"
          :failed-indexes="enSide === 'source' ? failedIndexes : []"
          :editable="!translating"
          :chinese-sentences="chineseSentences"
          v-model:html="enHtml"
          v-model:sentence-index="enSentenceIndex"
        />
      </section>
    </div>
  </div>
</template>
