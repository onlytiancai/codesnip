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
  }>(),
  { allowAi: true },
)

const emit = defineEmits<{
  /** 用新文本替换当前选区 */
  replace: [text: string]
  /** 用户关掉菜单 */
  close: []
}>()

/** 面板模式 */
type Panel =
  | { mode: 'candidates'; candidates: string[] }
  | { mode: 'tone' }
  | { mode: 'grammar'; text: string }
  | { mode: 'instruction' }
  | { mode: 'lookup'; word: string; entry: DictionaryEntry | null; missing: string }
  | { mode: 'result'; original: string; rewritten: string }

const panel = ref<Panel | null>(null)
const loading = ref(false)
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

// 选区变了（或者组件是新开的）→ 状态清零
watch(() => props.selection, resetState)

/** 统一的请求包装：管 loading / error / 取消。 */
async function run<T>(fn: (signal: AbortSignal) => Promise<T>): Promise<T | null> {
  abortInFlight()
  const ctl = new AbortController()
  controller.value = ctl
  loading.value = true
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

// ---- 候选词 / 短语 ----

async function openCandidates() {
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
  const word = s.text.trim()
  const result = await run((signal) => lookupWord(word, signal))
  if (result === null) return
  panel.value = {
    mode: 'lookup',
    word,
    entry: result.entry ?? null,
    missing: result.entry ? '' : (result.error ?? '词典里没有这个词'),
  }
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
  panel.value = null
  errorMsg.value = ''
}

/** Escape 逐层退出：面板 → 菜单 → 关闭 */
function onKeydown(e: KeyboardEvent) {
  if (e.key !== 'Escape') return
  e.stopPropagation()
  if (panel.value) {
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

// autoAction：工具栏按钮直接落到某个面板。
// 必须放在 setup 末尾——immediate 回调是同步跑的，而 runAction 会摸 sel / panel，
// 写在它们前面就还在 TDZ 里。
watch(
  () => props.autoAction,
  (action) => {
    if (action) void runAction(action)
  },
  { immediate: true },
)
</script>

<template>
  <!-- mousedown 阻止默认：否则点菜单会把 ProseMirror 的选区吃掉 -->
  <div class="ai-menu" role="dialog" :aria-label="`对「${selection.text}」的 AI 操作`" @mousedown.prevent>
    <!-- 头部 -->
    <header class="ai-menu__head">
      <span class="ai-menu__title" :title="selection.text">{{ selection.text }}</span>
      <button type="button" class="ai-menu__close" aria-label="关闭" @click="emit('close')">×</button>
    </header>

    <!-- ============ 主菜单 ============ -->
    <nav v-if="!panel" class="ai-menu__items">
      <button
        v-if="allowAi && selection.kind !== 'sentence'"
        type="button"
        class="ai-menu__item"
        :disabled="loading"
        @click="openCandidates"
      >
        {{ selection.kind === 'word' ? '🔄 替换词' : '🔄 替换短语' }}
      </button>

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

      <!-- 查词：本地词库，两个方向都有 -->
      <button
        v-if="selection.kind === 'word'"
        type="button"
        class="ai-menu__item"
        :disabled="loading"
        @click="runLookup"
      >
        📖 查词
      </button>

      <button
        v-if="allowAi && selection.kind !== 'phrase'"
        type="button"
        class="ai-menu__item"
        :disabled="loading"
        @click="runGrammarCheck"
      >
        🔍 检查语法
      </button>

      <button
        v-if="allowAi"
        type="button"
        class="ai-menu__item"
        :disabled="loading"
        @click="panel = { mode: 'instruction' }"
      >
        💬 自定义指令…
      </button>
    </nav>

    <!-- ============ 二级面板 ============ -->
    <div v-else class="ai-menu__panel">
      <!-- loading -->
      <div v-if="loading" class="ai-menu__status">模型生成中…</div>

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

          <!-- 词库只收词头：查 children 会命中 child -->
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

        <p v-else class="ai-menu__empty">{{ panel.missing }}</p>

        <button type="button" class="ai-menu__back" @click="backToMenu">返回</button>
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
  </div>
</template>
