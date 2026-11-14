<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { type ToneId } from '../constants/tones'
import { useTones } from '../composables/useTones'
import type { AiAction } from '../constants/aiActions'
import type { SelectionInfo } from '../composables/useSelectionKind'
import {
  fetchGrammarText,
  fetchInstructionRewrite,
  fetchSelectionCandidates,
  fetchSentenceRewrite,
  RewriteError,
  type SentenceOperation,
} from '../services/rewrite'
import { lookupWord, type DictionaryEntry } from '../services/dictionary'

/**
 * 跟随选区弹出的 AI 上下文菜单。
 *
 * 三层状态：
 *   menu    —— 顶部菜单，项随选区类型变化
 *   panel   —— 二级面板（候选列表 / 语气 / 语法结果 / 指令输入 / 词条 / 结果卡片）
 *   关闭    —— 父组件把 selection 置空即销毁
 *
 * 关键交互：整块区域在 mousedown 上 preventDefault，
 * 否则 ProseMirror 会先把选区吃掉，点「候选词」时 range 已经没了。
 *
 * `allowAi=false`（英译中模式）时，菜单里只剩「📖 查词」——
 * 那个方向只要翻译，不要改写/润色/语法那一套。
 */

/** 打开就直接落到某个面板——工具栏那排常驻按钮走这条路径。 */

const props = withDefaults(
  defineProps<{
    selection: SelectionInfo
    /** 选区所在句对应的中文原文，传给 API 消歧 */
    chineseSentence?: string
    /** 非空则一打开就直接进入该面板（跳过菜单） */
    autoAction?: AiAction | null
    /** 改写/语法/指令那几项要不要显示（英译中模式关掉） */
    allowAi?: boolean
    /**
     * 自定义指令跑出来的结果。
     *
     * 父组件（编辑器）负责跑指令（指令文本在服务器库里），
     * 跑完把结果塞进来 —— 菜单只负责显示结果卡片。
     */
    customResult?: { original: string; rewritten: string } | null
    /** 自定义指令还在跑 */
    customBusy?: boolean
    /** 自定义指令跑失败了 */
    customError?: string
    /**
     * 非空 = 直接查这个词（双击查词）。
     *
     * 菜单一打开就查好，用户不用再点一次「📖 查词」。
     */
    autoLookup?: string
  }>(),
  {
    allowAi: true,
    customResult: null,
    customBusy: false,
    customError: '',
    autoLookup: '',
  },
)

const emit = defineEmits<{
  /** 用新文本替换当前选区 */
  replace: [text: string]
  /** 用户关掉菜单 */
  close: []
  /**
   * 自定义指令的「继续调整」。
   *
   * 指令文本在服务器库里，父组件知道是哪一条 —— 由它去跑。
   */
  refine: [extra: string]
}>()

/** 顶栏的两个 tab（词 / 短语选中时才有） */
type Tab = 'candidates' | 'lookup'

/** 面板模式 */
type Panel =
  | { mode: 'candidates'; candidates: string[] }
  | { mode: 'tone' }
  | { mode: 'grammar'; text: string }
  | { mode: 'instruction' }
  | {
      mode: 'lookup'
      word: string
      entry: DictionaryEntry | null
      missing: string
      /** 词库里和它相近的词，点一下直接看那个词的解释 */
      suggestions: string[]
    }
  | { mode: 'result'; original: string; rewritten: string; custom?: boolean }

const panel = ref<Panel | null>(null)
/** 当前停在哪个 tab */
const tab = ref<Tab>('candidates')
const loading = ref(false)
/** loading 时显示什么 —— 查词是本地查库，别说「模型生成中…」 */
const loadingLabel = ref('模型生成中…')
const errorMsg = ref('')
const instruction = ref('')
/** 连续调整时的中间稿 */
const draft = ref('')

const controller = ref<AbortController | null>(null)

/** 语气列表（label 从后端拉，拉不到用内置的） */
const { tones } = useTones()

function abortInFlight() {
  controller.value?.abort()
  controller.value = null
}

/** 重置到一个干净的面板状态。 */
function resetState() {
  abortInFlight()
  panel.value = null
  loading.value = false
  errorMsg.value = ''
  instruction.value = ''
  draft.value = ''
}

/**
 * 词 / 短语选中 → 顶栏用两个 tab（其它候选单词 / 查词），直接落在第一个 tab。
 * 整句选中 → 还是原来的菜单项清单（改写语气 / 润色 / 精简 / 扩写）。
 *
 * 关掉 AI 能力（英译中）时也走菜单项清单 —— 那边只剩「📖 查词」。
 */
const showTabs = computed(() => props.allowAi && props.selection.kind !== 'sentence')

/** 统一的请求包装：管 loading / error / 取消。 */
async function run<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  label = '模型生成中…',
): Promise<T | null> {
  abortInFlight()
  const ctl = new AbortController()
  controller.value = ctl
  loading.value = true
  loadingLabel.value = label
  errorMsg.value = ''
  try {
    return await fn(ctl.signal)
  } catch (err) {
    // 取消不算错误，安静收场
    if (err instanceof DOMException && err.name === 'AbortError') return null
    errorMsg.value =
      err instanceof RewriteError
        ? `[${err.status}] ${err.message}`
        : err instanceof Error
          ? err.message
          : 'unknown error'
    return null
  } finally {
    loading.value = false
    if (controller.value === ctl) controller.value = null
  }
}

const sel = computed(() => props.selection)

/**
 * 菜单标题：查词面板显示「正在查的那个词」（点了相近词会换），
 * 其它面板显示选中的文本。
 */
const menuTitle = computed(() =>
  panel.value?.mode === 'lookup' ? panel.value.word : props.selection.text,
)

// ---- 候选词 / 短语 ----

/** 候选词请求的防抖：拖选时选区一直在变，等它稳下来再发。 */
const CANDIDATES_DELAY = 250
let candidatesTimer: ReturnType<typeof setTimeout> | undefined

/**
 * 打开候选词面板。
 *
 * 面板**先摆出来**（空列表 + loading），请求延后一点点再发 ——
 * 拖选的时候选区每挪一下就变一次，不防抖的话每变一次发一次请求。
 */
function openCandidates() {
  panel.value = { mode: 'candidates', candidates: [] }
  loading.value = true
  loadingLabel.value = '模型生成中…'
  errorMsg.value = ''
  abortInFlight()
  if (candidatesTimer) clearTimeout(candidatesTimer)
  candidatesTimer = setTimeout(() => {
    candidatesTimer = undefined
    void fetchCandidates()
  }, CANDIDATES_DELAY)
}

async function fetchCandidates() {
  const s = sel.value
  const cands = await run((signal) =>
    fetchSelectionCandidates(
      {
        text: s.text,
        kind: s.kind === 'word' ? 'word' : 'phrase',
        context: s.paragraph || undefined,
        sourceSentence: props.chineseSentence || undefined,
      },
      signal,
    ),
  )
  if (cands === null) return
  if (cands.length === 0) {
    errorMsg.value = '模型没有给出候选词'
    return
  }
  panel.value = { mode: 'candidates', candidates: cands }
}

/** 切顶栏的 tab。 */
function switchTab(next: Tab) {
  tab.value = next
  if (next === 'candidates') void openCandidates()
  else void runLookup()
}

// ---- 整句改写 ----

async function runSentenceRewrite(operation: SentenceOperation, tone?: ToneId) {
  const s = sel.value
  const original = s.kind === 'sentence' ? s.text : s.sentence
  const text = await run((signal) =>
    fetchSentenceRewrite(
      {
        sentence: original,
        operation,
        tone,
        context: s.paragraph || undefined,
        sourceSentence: props.chineseSentence || undefined,
      },
      signal,
    ),
  )
  if (text === null) return
  draft.value = text
  panel.value = { mode: 'result', original, rewritten: text }
}

// ---- 语法检查 ----

async function runGrammarCheck() {
  const s = sel.value
  const text = await run((signal) => fetchGrammarText(s.sentence, signal))
  if (text === null) return
  panel.value = { mode: 'grammar', text }
}

// ---- 自定义指令 ----

async function submitInstruction() {
  const instr = instruction.value.trim()
  if (!instr) return
  const s = sel.value

  const out = await run((signal) =>
    fetchInstructionRewrite(
      {
        text: s.text,
        instruction: instr,
        context: s.paragraph || undefined,
      },
      signal,
    ),
  )
  if (out === null) return
  draft.value = out
  panel.value = { mode: 'result', original: s.text, rewritten: out }
}

/** 在已有结果上继续调整：以当前草稿为新输入，跑同一（或新）指令。 */
async function refineInstruction() {
  const extra = instruction.value.trim()
  if (!extra || !draft.value) return

  // 自定义指令：指令文本在服务器库里，父组件知道是哪一条 —— 让它去跑
  if (panel.value?.mode === 'result' && panel.value.custom) {
    emit('refine', extra)
    return
  }

  const s = sel.value

  const out = await run((signal) =>
    fetchInstructionRewrite(
      {
        text: draft.value,
        instruction: extra,
        context: s.paragraph || undefined,
      },
      signal,
    ),
  )
  if (out === null) return
  panel.value = { mode: 'result', original: draft.value, rewritten: out }
}

/** autoAction → 具体动作。语气面板本身不发请求，等用户点一个语气。 */
function runAction(action: AiAction): void | Promise<void> {
  switch (action) {
    case 'candidates':
      return openCandidates()
    case 'tone':
      panel.value = { mode: 'tone' }
      return
    case 'improve':
      return runSentenceRewrite('improve')
    case 'shorten':
      return runSentenceRewrite('shorten')
    case 'expand':
      return runSentenceRewrite('expand')
    case 'grammar':
      return runGrammarCheck()
    case 'instruction':
      instruction.value = ''
      panel.value = { mode: 'instruction' }
      return
    case 'lookup':
      return runLookup()
  }
}

// ---- 查词（本地词库，不调模型）----

async function runLookup() {
  const s = sel.value
  await lookupWordInto(s.text.trim())
}

/**
 * 查一个词并把结果铺到面板上。
 * 查不到时把后端给的相近词一起带上面板，用户点一下就能看那个词。
 */
async function lookupWordInto(word: string) {
  const result = await run((signal) => lookupWord(word, signal), '查词中…')
  if (result === null) return
  panel.value = {
    mode: 'lookup',
    word,
    entry: result.entry ?? null,
    missing: result.entry ? '' : (result.error ?? '词典里没有这个词'),
    suggestions: result.entry ? [] : (result.suggestions ?? []),
  }
}

/** 点「相近的词」→ 换成查那个词，面板内容跟着换。 */
async function openSuggestion(word: string) {
  await lookupWordInto(word)
}

// ---- 结果卡片 ----

function accept() {
  const p = panel.value
  if (p?.mode !== 'result') return
  emit('replace', p.rewritten)
  emit('close')
}

function reject() {
  emit('close')
}

function backToMenu() {
  abortInFlight()
  errorMsg.value = ''
  // tab 模式下「返回」= 回到「其它候选单词」那个 tab
  if (showTabs.value) {
    switchTab('candidates')
    return
  }
  panel.value = null
}

/** Escape 逐层退出：面板 → 菜单 → 关闭 */
function onKeydown(e: KeyboardEvent) {
  if (e.key !== 'Escape') return
  e.stopPropagation()
  // tab 模式下面板一直开着，没有「上一层」可退 → 直接关掉
  if (panel.value && !showTabs.value) {
    backToMenu()
    return
  }
  emit('close')
}

// 菜单被 Teleport 到了 body，keydown 不会冒泡到编辑器那层，
// 所以自己挂在 window 上收 Escape（面板里 textarea 获焦时也要能退）。
onMounted(() => window.addEventListener('keydown', onKeydown))
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  abortInFlight()
})

// 面板里有输入框时自动聚焦
const instructionInput = ref<HTMLTextAreaElement | null>(null)
watch(instructionInput, (el) => el?.focus())

// 选区变了（或者组件是新开的）→ 状态清零；词/短语直接落在「其它候选单词」。
//
// 两个 watcher 都放 setup 末尾：immediate 回调是同步跑的，
// 选区那个会摸 candidatesTimer、autoAction 那个会摸 sel / panel ——
// 写在这些声明前面就还在 TDZ 里。
//
// 顺序也有讲究：选区的 watcher 要在 autoAction 前面 —— 它会 resetState()
// （顺手掐掉在飞的请求），反过来会把 autoAction 刚发出去的请求掐掉。
watch(
  () => props.selection,
  () => {
    resetState()
    tab.value = 'candidates'
    // 双击查词：直接查词，不跑候选词（省一次模型调用）
    if (showTabs.value && !props.autoLookup) void openCandidates()
  },
  { immediate: true },
)

// autoAction：工具栏按钮直接落到某个面板。
watch(
  () => props.autoAction,
  (action) => {
    if (action) void runAction(action)
  },
  { immediate: true },
)

// 双击查词：直接落到「查词」面板。
//
// ⚠️ 必须排在选区那个 watcher 后面 —— 选区 watcher 会 resetState()
// （顺手 abort 在飞的请求），反过来会把这里刚发起的查词掐掉。
watch(
  () => props.autoLookup,
  (word) => {
    if (!word) return
    tab.value = 'lookup'
    void lookupWordInto(word)
  },
  { immediate: true },
)

// 自定义指令的结果：父组件跑完塞进来 → 直接落到结果卡片。
//
// ⚠️ 必须排在选区那个 watcher 后面 —— 选区 watcher 会 resetState()，
// 把面板清干净；结果 watcher 得在它之后把卡片摆出来。
watch(
  () => props.customResult,
  (r) => {
    if (!r) return
    draft.value = r.rewritten
    panel.value = {
      mode: 'result',
      original: r.original,
      rewritten: r.rewritten,
      custom: true,
    }
  },
  { immediate: true },
)
</script>

<template>
  <!-- mousedown 阻止默认：否则点菜单会把 ProseMirror 的选区吃掉 -->
  <div class="ai-menu" role="dialog" :aria-label="`对「${selection.text}」的 AI 操作`" @mousedown.prevent>
    <!-- 头部 -->
    <header class="ai-menu__head">
      <span class="ai-menu__title" :title="menuTitle">{{ menuTitle }}</span>
      <button type="button" class="ai-menu__close" aria-label="关闭" @click="emit('close')">×</button>
    </header>

    <!-- ============ 顶栏 tab（词 / 短语选中） ============ -->
    <div v-if="showTabs" class="ai-menu__tabs" role="tablist">
      <button
        type="button"
        class="ai-menu__tab"
        :class="{ 'ai-menu__tab--active': tab === 'candidates' }"
        role="tab"
        :aria-selected="tab === 'candidates'"
        @click="switchTab('candidates')"
      >
        其它候选单词
      </button>
      <button
        type="button"
        class="ai-menu__tab"
        :class="{ 'ai-menu__tab--active': tab === 'lookup' }"
        role="tab"
        :aria-selected="tab === 'lookup'"
        @click="switchTab('lookup')"
      >
        查词
      </button>
    </div>

    <!-- ============ 菜单项清单（整句 / 关掉 AI 能力时） ============ -->
    <nav
      v-if="!showTabs && !panel && !customBusy && !customError && !loading"
      class="ai-menu__items"
    >
      <template v-if="allowAi && selection.kind === 'sentence'">
        <button
          type="button"
          class="ai-menu__item ai-menu__item--sub"
          :disabled="loading"
          @click="panel = { mode: 'tone' }"
        >
          🎨 改写语气 <span class="ai-menu__caret">▸</span>
        </button>
        <button type="button" class="ai-menu__item" :disabled="loading" @click="runSentenceRewrite('improve')">
          ✏️ 润色
        </button>
        <button type="button" class="ai-menu__item" :disabled="loading" @click="runSentenceRewrite('shorten')">
          ✂️ 精简
        </button>
        <button type="button" class="ai-menu__item" :disabled="loading" @click="runSentenceRewrite('expand')">
          ➕ 扩写
        </button>
      </template>

      <!-- 查词：本地词库（tab 模式下它在顶栏，这里是关掉 AI 能力时的入口） -->
      <button
        v-if="selection.kind === 'word'"
        type="button"
        class="ai-menu__item"
        :disabled="loading"
        @click="runLookup"
      >
        📖 查词
      </button>
    </nav>

    <!-- ============ 二级面板 ============ -->

    <!-- 自定义指令：正在跑 / 跑失败了 -->
    <div v-if="customBusy" class="ai-menu__panel">
      <div class="ai-menu__status">指令生成中…</div>
    </div>
    <div v-else-if="customError" class="ai-menu__panel">
      <div class="ai-menu__error">
        <p>{{ customError }}</p>
        <button type="button" @click="emit('close')">关闭</button>
      </div>
    </div>

    <!-- 请求在飞：先显示状态，别闪一下菜单项 -->
    <div v-else-if="loading" class="ai-menu__panel">
      <div class="ai-menu__status">{{ loadingLabel }}</div>
    </div>

    <div v-else-if="panel" class="ai-menu__panel">
      <!-- loading -->
      <div v-if="loading" class="ai-menu__status">{{ loadingLabel }}</div>

      <!-- 错误 -->
      <div v-else-if="errorMsg" class="ai-menu__error">
        <p>{{ errorMsg }}</p>
        <button type="button" @click="backToMenu">返回</button>
      </div>

      <!-- 候选列表 -->
      <ul v-else-if="panel.mode === 'candidates'" class="ai-menu__list">
        <li v-for="c in panel.candidates" :key="c">
          <button type="button" class="ai-menu__cand" @click="emit('replace', c); emit('close')">
            {{ c }}
          </button>
        </li>
      </ul>

      <!-- 语气选择 -->
      <ul v-else-if="panel.mode === 'tone'" class="ai-menu__list">
        <li v-for="t in tones" :key="t.id">
          <button type="button" class="ai-menu__cand" @click="runSentenceRewrite('rewrite', t.id)">
            {{ t.label }}
          </button>
        </li>
      </ul>

      <!-- 语法结果：模型写的原文，只提示不自动改 -->
      <div v-else-if="panel.mode === 'grammar'" class="ai-menu__status">
        <pre class="ai-menu__grammar">{{ panel.text }}</pre>
      </div>

      <!-- 查词结果：本地词库 -->
      <div v-else-if="panel.mode === 'lookup'" class="ai-menu__status ai-menu__dict">
        <template v-if="panel.entry">
          <p class="ai-menu__dict-head">
            <strong>{{ panel.entry.word }}</strong>
            <span v-if="panel.entry.ukphone || panel.entry.usphone" class="ai-menu__dict-phonetic">
              /{{ panel.entry.ukphone || panel.entry.usphone }}/
            </span>
          </p>

          <!-- 词库只收词头：查 abandons 会命中 abandon -->
          <p
            v-if="panel.entry.word.toLowerCase() !== panel.word.toLowerCase()"
            class="ai-menu__dict-note"
          >
            词库里收的是「{{ panel.entry.word }}」
          </p>

          <ul v-if="panel.entry.trans.length" class="ai-menu__dict-list">
            <li v-for="(t, i) in panel.entry.trans" :key="i">
              <span v-if="t.pos" class="ai-menu__dict-pos">{{ t.pos }}</span>
              {{ t.cn }}
            </li>
          </ul>

          <ul v-if="panel.entry.sentences.length" class="ai-menu__dict-list">
            <li v-for="(s, i) in panel.entry.sentences" :key="`s${i}`">
              <span class="ai-menu__dict-en">{{ s.en }}</span>
              <span v-if="s.cn" class="ai-menu__dict-cn">{{ s.cn }}</span>
            </li>
          </ul>

          <ul v-if="panel.entry.phrases.length" class="ai-menu__dict-list">
            <li v-for="(p, i) in panel.entry.phrases" :key="`p${i}`">
              <span class="ai-menu__dict-en">{{ p.en }}</span>
              <span v-if="p.cn" class="ai-menu__dict-cn">{{ p.cn }}</span>
            </li>
          </ul>

          <ul v-if="panel.entry.synonyms.length" class="ai-menu__dict-list">
            <li v-for="(s, i) in panel.entry.synonyms" :key="`y${i}`">
              <span v-if="s.pos" class="ai-menu__dict-pos">{{ s.pos }}</span>
              <span v-if="s.tran">{{ s.tran }}</span>
              <span v-if="s.words.length" class="ai-menu__dict-words">
                （{{ s.words.join(' / ') }}）
              </span>
            </li>
          </ul>

          <p v-if="panel.entry.antonyms.length" class="ai-menu__dict-line">
            <span class="ai-menu__dict-pos">反义</span>
            {{ panel.entry.antonyms.join(' / ') }}
          </p>

          <p v-if="panel.entry.mnemonic" class="ai-menu__dict-line">
            <span class="ai-menu__dict-pos">记忆</span>
            {{ panel.entry.mnemonic }}
          </p>
        </template>

        <template v-else>
          <p class="ai-menu__empty">{{ panel.missing }}</p>

          <!-- 查不到 → 列出词库里相近的词，点一下直接看那个词的解释 -->
          <div v-if="panel.suggestions.length" class="ai-menu__dict-suggest">
            <p class="ai-menu__dict-suggest-title">词库里相近的词：</p>
            <div class="ai-menu__chips">
              <button
                v-for="w in panel.suggestions"
                :key="w"
                type="button"
                class="ai-menu__chip ai-menu__chip--word"
                :disabled="loading"
                @click="openSuggestion(w)"
              >
                {{ w }}
              </button>
            </div>
          </div>
        </template>

        <button v-if="!showTabs" type="button" class="ai-menu__back" @click="backToMenu">返回</button>
      </div>

      <!-- 指令输入 -->
      <div v-else-if="panel.mode === 'instruction'" class="ai-menu__status">
        <textarea
          ref="instructionInput"
          v-model="instruction"
          class="ai-menu__textarea"
          rows="2"
          placeholder="例如：Make it more academic. / 更口语一点"
        ></textarea>
        <div class="ai-menu__chips">
          <button
            v-for="s in ['Make it more academic.', 'Make it sound more natural.', 'Simplify the sentence.', 'Make it more persuasive.', 'Rewrite it in a formal tone.']"
            :key="s"
            type="button"
            class="ai-menu__chip"
            @click="instruction = s"
          >
            {{ s }}
          </button>
        </div>
        <button type="button" :disabled="!instruction.trim()" @click="submitInstruction">生成</button>
      </div>

      <!-- 结果卡片：接受 / 拒绝 -->
      <div v-else-if="panel.mode === 'result'" class="ai-menu__status">
        <p class="ai-menu__original">{{ panel.original }}</p>
        <p class="ai-menu__rewritten">{{ panel.rewritten }}</p>
        <div class="ai-menu__actions">
          <button type="button" class="ai-menu__accept" @click="accept">✓ 接受</button>
          <button type="button" @click="reject">✕ 拒绝</button>
        </div>
        <details class="ai-menu__refine">
          <summary>继续调整</summary>
          <textarea
            v-model="instruction"
            class="ai-menu__textarea"
            rows="2"
            placeholder="再补一句要求，例如：再短一点"
          ></textarea>
          <button type="button" :disabled="!instruction.trim()" @click="refineInstruction">继续生成</button>
        </details>
      </div>
    </div>

    <!-- tab 模式下 panel 还没摆出来的瞬间（正常只有一帧） -->
    <div v-else-if="showTabs" class="ai-menu__panel">
      <div class="ai-menu__status">{{ loadingLabel }}</div>
    </div>
  </div>
</template>
