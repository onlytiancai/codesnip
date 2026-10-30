<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import {
  copyPrompts,
  dryRunPrompt,
  fetchPromptSet,
  fetchVersion,
  fetchVersions,
  rollbackPrompt,
  savePrompt,
  saveTones,
  type DryRunResult,
  type PromptSet,
  type PromptTemplate,
  type PromptVersionMeta,
  type ProviderConfig,
  type ToneConfig,
} from '../services/settings'

/**
 * 提示词编辑（按 provider）。
 *
 * 左边选能力，右边改模板；每个模板有独立的版本历史，能回滚；
 * 「试运行」用**正在编辑的这个 provider** 真跑一次，看实际发出去什么、
 * 模型回了什么、解析出来是什么。
 */

const props = defineProps<{
  providers: ProviderConfig[]
  activeProviderId: string
}>()

const TONES_TAB = '__tones__'

const providerId = ref(props.activeProviderId)
const promptSet = ref<PromptSet | null>(null)
const selected = ref<string>('translate')
const draft = ref<PromptTemplate | null>(null)
const schemaText = ref('')
const note = ref('')
const tones = ref<ToneConfig[]>([])
const versions = ref<PromptVersionMeta[]>([])

const busy = ref(false)
const error = ref('')
const message = ref('')

const dryRunVars = ref<Record<string, string>>({})
const dryRunVarKeys = computed(() => Object.keys(dryRunVars.value))
const dryRunResult = ref<DryRunResult | null>(null)
const dryRunBusy = ref(false)

// ---- 载入 ----

async function loadPromptSet(id: string) {
  busy.value = true
  error.value = ''
  try {
    const set = await fetchPromptSet(id)
    promptSet.value = set
    tones.value = set.tones.map((t) => ({ ...t }))
    const cap = set.templates.find((t) => t.id === selected.value)
    if (cap) selectTemplate(cap)
    else if (set.templates[0]) selectTemplate(set.templates[0])
    await loadVersions()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

async function loadVersions() {
  if (selected.value === TONES_TAB) {
    versions.value = []
    return
  }
  try {
    const res = await fetchVersions(providerId.value, selected.value)
    versions.value = res.versions
  } catch {
    versions.value = []
  }
}

function selectTemplate(t: PromptTemplate) {
  draft.value = { ...t, jsonSchema: t.jsonSchema ? { ...t.jsonSchema } : undefined }
  schemaText.value = t.jsonSchema ? JSON.stringify(t.jsonSchema, null, 2) : ''
  note.value = ''
  dryRunResult.value = null
  const sample = promptSet.value?.samples?.[t.id] ?? {}
  dryRunVars.value = { ...sample }
}

watch(providerId, (id) => {
  void loadPromptSet(id)
})

// 组件挂载时 providerId 已经定了 —— watch 不会触发，得自己先载一次，
// 否则左边的能力列表是空的（只有「语气表」）。
onMounted(() => {
  void loadPromptSet(providerId.value)
})

watch(selected, () => {
  const set = promptSet.value
  if (!set) return
  if (selected.value === TONES_TAB) {
    draft.value = null
    tones.value = set.tones.map((t) => ({ ...t }))
    void loadVersions()
    return
  }
  const t = set.templates.find((x) => x.id === selected.value)
  if (t) selectTemplate(t)
  void loadVersions()
})

// provider 切换时（比如别的 tab 里改了生效的 provider），跟着切
watch(
  () => props.activeProviderId,
  (id) => {
    if (id && id !== providerId.value) providerId.value = id
  },
)

const capabilities = computed(() => promptSet.value?.templates ?? [])
const variables = computed(() => {
  const id = selected.value
  if (id === TONES_TAB) return []
  return promptSet.value?.variables?.[id] ?? []
})

const isJsonMode = computed(() => draft.value?.responseMode === 'json')

// ---- 保存 ----

async function save() {
  const d = draft.value
  if (!d) return
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    const template: PromptTemplate = { ...d }
    if (template.responseMode === 'json') {
      try {
        template.jsonSchema = JSON.parse(schemaText.value)
      } catch {
        error.value = 'JSON Schema 不是合法 JSON'
        return
      }
    } else {
      delete template.jsonSchema
    }
    const res = await savePrompt(providerId.value, selected.value, template, note.value || undefined)
    message.value = `已保存为 v${res.version}`
    note.value = ''
    await loadPromptSet(providerId.value)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

async function saveToneTable() {
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    await saveTones(providerId.value, tones.value)
    message.value = '语气表已保存'
    await loadPromptSet(providerId.value)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

// ---- 版本历史 ----

async function rollback(v: number) {
  if (!window.confirm(`回滚到 v${v}？会把它复制成一个新版本（历史只增不减）。`)) {
    return
  }
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    const res = await rollbackPrompt(providerId.value, selected.value, v)
    message.value = `已回滚到 v${v}（新版本 v${res.version}）`
    await loadPromptSet(providerId.value)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

async function viewVersion(v: number) {
  busy.value = true
  try {
    const rec = await fetchVersion(providerId.value, selected.value, v)
    draft.value = { ...rec.template }
    schemaText.value = rec.template.jsonSchema
      ? JSON.stringify(rec.template.jsonSchema, null, 2)
      : ''
    message.value = `正在看 v${v} 的内容（还没保存）`
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

// ---- 复制 ----

async function copyFrom() {
  const others = props.providers.filter((p) => p.id !== providerId.value)
  if (!others.length) {
    message.value = '没有别的 provider 可以复制'
    return
  }
  const from = window.prompt(
    `从哪个 provider 复制？\n${others.map((p) => p.id).join(' / ')}`,
    others[0].id,
  )
  if (!from) return
  busy.value = true
  error.value = ''
  try {
    await copyPrompts(providerId.value, from, selected.value)
    message.value = `已从「${from}」复制`
    await loadPromptSet(providerId.value)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

// ---- 试运行 ----

async function dryRun() {
  dryRunBusy.value = true
  dryRunResult.value = null
  error.value = ''
  try {
    const payload: Record<string, unknown> = { vars: dryRunVars.value }
    if (draft.value) {
      payload.template = {
        id: draft.value.id,
        label: draft.value.label,
        system: draft.value.system,
        user: draft.value.user,
        responseMode: draft.value.responseMode,
        ...(isJsonMode.value ? { jsonSchema: JSON.parse(schemaText.value) } : {}),
      }
    }
    dryRunResult.value = await dryRunPrompt(providerId.value, selected.value, payload)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    dryRunBusy.value = false
  }
}

function insertVar(name: string) {
  const d = draft.value
  if (!d) return
  const token = `{{${name}}}`
  d.user = d.user ? `${d.user} ${token}` : token
}
</script>

<template>
  <section class="prompts">
    <div class="prompts__bar">
      <label class="field field--inline">
        <span class="field__label">Provider</span>
        <select v-model="providerId" class="field__input">
          <option v-for="p in providers" :key="p.id" :value="p.id">
            {{ p.label }}（{{ p.id }}）{{ p.id === activeProviderId ? ' · 使用中' : '' }}
          </option>
        </select>
      </label>
      <span class="prompts__hint">
        每个 provider 的提示词完全独立 —— 这里改的只影响当前选中的 provider。
      </span>
    </div>

    <p v-if="error" class="prompts__error">{{ error }}</p>
    <p v-if="message" class="prompts__ok">{{ message }}</p>

    <div class="prompts__body">
      <!-- 左：能力列表 -->
      <nav class="prompts__nav">
        <button
          v-for="t in capabilities"
          :key="t.id"
          class="prompts__nav-item"
          :class="{ 'prompts__nav-item--on': selected === t.id }"
          @click="selected = t.id"
        >
          <span>{{ t.label }}</span>
          <span class="prompts__nav-ver">
            {{ t.version ? `v${t.version}` : '默认' }}
          </span>
        </button>
        <button
          class="prompts__nav-item"
          :class="{ 'prompts__nav-item--on': selected === TONES_TAB }"
          @click="selected = TONES_TAB"
        >
          <span>语气表</span>
          <span class="prompts__nav-ver">8 种</span>
        </button>
      </nav>

      <!-- 右：编辑器 -->
      <div class="prompts__editor">
        <!-- 语气表 -->
        <template v-if="selected === TONES_TAB">
          <h3 class="prompts__title">语气表</h3>
          <p class="prompts__desc">
            语气 id 固定（前端下拉要用），但每个 provider 的中文 label 和英文
            guidance 可以各自改 —— guidance 是要拼进 prompt 正文的。
          </p>
          <div
            v-for="t in tones"
            :key="t.id"
            class="prompts__tone"
          >
            <div class="prompts__tone-head">
              <code>{{ t.id }}</code>
              <input v-model="t.label" class="field__input field__input--sm" placeholder="中文 label" />
            </div>
            <textarea
              v-model="t.guidance"
              class="field__input field__textarea"
              rows="2"
            />
          </div>
          <div class="prompts__actions">
            <button class="btn" :disabled="busy" @click="saveToneTable">保存语气表</button>
          </div>
        </template>

        <!-- 模板编辑 -->
        <template v-else-if="draft">
          <h3 class="prompts__title">{{ draft.label }}</h3>

          <div class="prompts__vars">
            <span class="prompts__vars-label">可用变量：</span>
            <button
              v-for="v in variables"
              :key="v"
              class="prompts__var"
              @click="insertVar(v)"
            >
              {{ v }}
            </button>
          </div>

          <label class="field">
            <span class="field__label">System 模板</span>
            <textarea v-model="draft.system" class="field__input field__textarea" rows="10" />
          </label>

          <label class="field">
            <span class="field__label">User 模板</span>
            <textarea v-model="draft.user" class="field__input field__textarea" rows="8" />
          </label>

          <div class="prompts__modes">
            <span class="field__label">输出模式</span>
            <label class="prompts__mode">
              <input v-model="draft.responseMode" type="radio" value="text" />
              纯文本
            </label>
            <label class="prompts__mode">
              <input v-model="draft.responseMode" type="radio" value="json" />
              格式化 JSON
            </label>
            <span class="field__hint">
              JSON 模式会让 provider 用 schema 强制输出，解析出来的形状和纯文本一样
            </span>
          </div>

          <label v-if="isJsonMode" class="field">
            <span class="field__label">JSON Schema</span>
            <textarea v-model="schemaText" class="field__input field__textarea" rows="10" />
          </label>

          <div class="prompts__actions">
            <input v-model="note" class="field__input field__input--sm" placeholder="备注（可选）" />
            <button class="btn" :disabled="busy" @click="save">保存</button>
            <button class="btn btn--ghost" :disabled="busy" @click="copyFrom">从别的 provider 复制</button>
          </div>

          <!-- 版本历史 -->
          <div class="prompts__versions">
            <h4 class="prompts__subtitle">版本历史</h4>
            <p v-if="!versions.length" class="prompts__desc">
              还没改过 —— 用的是内置默认提示词。
            </p>
            <ul v-else class="prompts__version-list">
              <li v-for="v in versions.slice().reverse()" :key="v.version" class="prompts__version">
                <span class="prompts__version-no">v{{ v.version }}</span>
                <span class="prompts__version-note">{{ v.note || '（无备注）' }}</span>
                <span class="prompts__version-time">{{ new Date(v.savedAt).toLocaleString() }}</span>
                <button class="btn btn--ghost btn--sm" @click="viewVersion(v.version)">查看</button>
                <button class="btn btn--ghost btn--sm" @click="rollback(v.version)">回滚</button>
              </li>
            </ul>
          </div>

          <!-- 试运行 -->
          <div class="prompts__dry">
            <h4 class="prompts__subtitle">试运行</h4>
            <p class="prompts__desc">
              用<strong>当前这个 provider</strong>真跑一次 —— 会真的消耗 API 额度。
            </p>
            <div class="prompts__dry-inputs">
              <label v-for="key in dryRunVarKeys" :key="key" class="field">
                <span class="field__label">{{ key }}</span>
                <input v-model="dryRunVars[key]" class="field__input field__input--sm" />
              </label>
            </div>
            <button class="btn" :disabled="dryRunBusy" @click="dryRun">
              {{ dryRunBusy ? '运行中…' : '运行' }}
            </button>

            <div v-if="dryRunResult" class="prompts__dry-result">
              <div v-if="dryRunResult.error" class="prompts__error">
                调用失败：{{ dryRunResult.error }}
              </div>
              <template v-else>
                <h5>实际发出的 system</h5>
                <pre class="prompts__pre">{{ dryRunResult.system }}</pre>
                <h5>实际发出的 user</h5>
                <pre class="prompts__pre">{{ dryRunResult.user }}</pre>
                <h5>模型原始输出</h5>
                <pre class="prompts__pre">{{ dryRunResult.raw }}</pre>
                <h5>
                  解析结果
                  <span v-if="dryRunResult.parseError" class="prompts__error">
                    （解析失败：{{ dryRunResult.parseError }}）
                  </span>
                </h5>
                <pre class="prompts__pre">{{ dryRunResult.parsed }}</pre>
              </template>
            </div>
          </div>
        </template>

        <p v-else class="prompts__desc">加载中…</p>
      </div>
    </div>
  </section>
</template>
