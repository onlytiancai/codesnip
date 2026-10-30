<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  deleteProvider,
  fetchModels,
  saveProvider,
  setActiveProvider,
  testProvider,
  type ProviderConfig,
  type ProviderKind,
  type StructuredOutputMode,
} from '../services/settings'

/**
 * Provider 列表 + 编辑表单。
 *
 * 每个 provider 自带 baseUrl / model / key / 结构化输出策略 ——
 * 换 provider 就是换一整套「怎么跟模型说话」的参数。
 */

defineProps<{
  providers: ProviderConfig[]
  activeProviderId: string
}>()

const emit = defineEmits<{ (e: 'changed'): void }>()

const KIND_LABEL: Record<ProviderKind, string> = {
  ollama: 'Ollama',
  openai: 'OpenAI 兼容',
  anthropic: 'Anthropic',
}

const STRUCTURED_LABEL: Record<StructuredOutputMode, string> = {
  schema: '强制 JSON Schema',
  object: '只强制 JSON 对象',
  prompt: '只靠提示词要求',
  off: '不要求 JSON',
}

interface Draft {
  id: string
  label: string
  kind: ProviderKind
  baseUrl: string
  model: string
  apiKeyEnv: string
  structuredOutput: StructuredOutputMode
  maxTokens: number | undefined
  maxTokensParam: 'max_tokens' | 'max_completion_tokens'
  temperature: number | undefined
  timeoutMs: number | undefined
}

const editing = ref<Draft | null>(null)
const editingOriginalId = ref<string | null>(null)
const busy = ref(false)
const message = ref('')
const error = ref('')
const modelOptions = ref<string[]>([])

const isNew = computed(() => editingOriginalId.value === null)

function blankDraft(): Draft {
  return {
    id: '',
    label: '',
    kind: 'openai',
    baseUrl: '',
    model: '',
    apiKeyEnv: '',
    structuredOutput: 'schema',
    maxTokens: undefined,
    maxTokensParam: 'max_tokens',
    temperature: undefined,
    timeoutMs: undefined,
  }
}

function startNew() {
  editing.value = blankDraft()
  editingOriginalId.value = null
  modelOptions.value = []
  message.value = ''
  error.value = ''
}

function startEdit(p: ProviderConfig) {
  editing.value = {
    id: p.id,
    label: p.label,
    kind: p.kind,
    baseUrl: p.baseUrl,
    model: p.model,
    apiKeyEnv: p.apiKeyEnv ?? '',
    structuredOutput: p.structuredOutput,
    maxTokens: p.maxTokens,
    maxTokensParam: p.maxTokensParam ?? 'max_tokens',
    temperature: p.temperature,
    timeoutMs: p.timeoutMs,
  }
  editingOriginalId.value = p.id
  modelOptions.value = []
  message.value = ''
  error.value = ''
}

function cancel() {
  editing.value = null
  editingOriginalId.value = null
  message.value = ''
  error.value = ''
}

async function save() {
  const d = editing.value
  if (!d) return
  if (!d.id.trim() || !d.label.trim() || !d.model.trim()) {
    error.value = 'id、名称、模型都是必填的'
    return
  }
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    const body: Record<string, unknown> = {
      id: d.id.trim(),
      label: d.label.trim(),
      kind: d.kind,
      baseUrl: d.baseUrl.trim(),
      model: d.model.trim(),
      structuredOutput: d.structuredOutput,
    }
    // 环境变量名：留空 = 这个 provider 不读环境变量
    if (d.apiKeyEnv.trim()) body.apiKeyEnv = d.apiKeyEnv.trim()
    if (d.maxTokens !== undefined) body.maxTokens = d.maxTokens
    if (d.maxTokensParam) body.maxTokensParam = d.maxTokensParam
    if (d.temperature !== undefined) body.temperature = d.temperature
    if (d.timeoutMs !== undefined) body.timeoutMs = d.timeoutMs

    await saveProvider(body as never)
    message.value = isNew.value
      ? '已新增。提示词已用内置默认值初始化，可到「提示词」页单独改。'
      : '已保存'
    editing.value = null
    editingOriginalId.value = null
    emit('changed')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

async function activate(id: string) {
  busy.value = true
  error.value = ''
  try {
    await setActiveProvider(id)
    message.value = '已切换生效的 provider'
    emit('changed')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

async function remove(id: string) {
  if (!window.confirm(`确定删除 provider「${id}」？它的提示词和历史也会一起删掉。`)) {
    return
  }
  busy.value = true
  error.value = ''
  try {
    await deleteProvider(id)
    message.value = '已删除'
    emit('changed')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

async function test(p: ProviderConfig) {
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    const res = await testProvider(p.id)
    message.value = `连接成功：${res.sample ?? '(空响应)'}`
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

async function loadModels() {
  const d = editing.value
  if (!d) return
  busy.value = true
  error.value = ''
  try {
    const res = await fetchModels(d.id.trim())
    modelOptions.value = res.models
    if (!res.models.length) {
      message.value = '这个 provider 没有返回任何模型'
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <section class="providers">
    <div class="providers__bar">
      <button class="btn" @click="startNew">+ 新增供应商</button>
      <span class="providers__hint">
        每个 provider 自带一套提示词 —— 换模型就换一套「怎么跟它说话」的规则。
      </span>
    </div>

    <p v-if="error" class="providers__error">{{ error }}</p>
    <p v-if="message" class="providers__ok">{{ message }}</p>

    <ul class="providers__list">
      <li
        v-for="p in providers"
        :key="p.id"
        class="providers__item"
        :class="{ 'providers__item--active': p.id === activeProviderId }"
      >
        <div class="providers__main">
          <div class="providers__title">
            <strong>{{ p.label }}</strong>
            <span class="providers__badge">{{ KIND_LABEL[p.kind] }}</span>
            <span v-if="p.id === activeProviderId" class="providers__badge providers__badge--on">
              使用中
            </span>
            <span v-if="p.hasKey" class="providers__badge providers__badge--key">key 已配置</span>
          </div>
          <div class="providers__meta">
            <span>{{ p.model }}</span>
            <span v-if="p.baseUrl" class="providers__url">{{ p.baseUrl }}</span>
            <span>{{ STRUCTURED_LABEL[p.structuredOutput] }}</span>
          </div>
        </div>
        <div class="providers__actions">
          <button
            v-if="p.id !== activeProviderId"
            class="btn btn--ghost"
            :disabled="busy"
            @click="activate(p.id)"
          >
            设为生效
          </button>
          <button class="btn btn--ghost" :disabled="busy" @click="startEdit(p)">
            编辑
          </button>
          <button class="btn btn--ghost" :disabled="busy" @click="test(p)">
            测试连接
          </button>
          <button
            class="btn btn--ghost btn--danger"
            :disabled="busy || p.id === activeProviderId"
            @click="remove(p.id)"
          >
            删除
          </button>
        </div>
      </li>
    </ul>

    <!-- 编辑表单 -->
    <div v-if="editing" class="providers__form">
      <h3 class="providers__form-title">
        {{ isNew ? '新增供应商' : `编辑「${editing.label}」` }}
      </h3>

      <div class="providers__grid">
        <label class="field">
          <span class="field__label">ID</span>
          <input v-model="editing.id" class="field__input" :disabled="!isNew" placeholder="openai-main" />
          <span class="field__hint">只能用字母、数字、_ . -</span>
        </label>

        <label class="field">
          <span class="field__label">名称</span>
          <input v-model="editing.label" class="field__input" placeholder="OpenAI" />
        </label>

        <label class="field">
          <span class="field__label">类型</span>
          <select v-model="editing.kind" class="field__input">
            <option value="ollama">Ollama（本地）</option>
            <option value="openai">OpenAI 兼容（含 DeepSeek / Moonshot / vLLM）</option>
            <option value="anthropic">Anthropic（Claude）</option>
          </select>
        </label>

        <label class="field">
          <span class="field__label">Base URL</span>
          <input
            v-model="editing.baseUrl"
            class="field__input"
            placeholder="https://api.openai.com/v1"
          />
          <span class="field__hint">Ollama 填 host（http://127.0.0.1:11434）</span>
        </label>

        <label class="field">
          <span class="field__label">模型</span>
          <input v-model="editing.model" class="field__input" placeholder="gpt-4o-mini" />
          <span class="field__hint">
            填好了 id 再点「拉取模型列表」可以自动填
          </span>
        </label>

        <label class="field">
          <span class="field__label">环境变量名</span>
          <input
            v-model="editing.apiKeyEnv"
            class="field__input"
            placeholder="MINIMAX_ANTHROPIC_KEY"
            autocomplete="off"
          />
          <span class="field__hint">
            key 写在 server/.env 里（不进 git），这里只填变量名。留空 = 不用 key。
          </span>
        </label>

        <label class="field">
          <span class="field__label">结构化输出</span>
          <select v-model="editing.structuredOutput" class="field__input">
            <option value="schema">强制 JSON Schema</option>
            <option value="object">只强制 JSON 对象</option>
            <option value="prompt">只靠提示词要求</option>
            <option value="off">不要求 JSON</option>
          </select>
          <span class="field__hint">
            端点不认时会自动降级：schema → object → 纯提示词
          </span>
        </label>

        <label class="field">
          <span class="field__label">max tokens</span>
          <input
            v-model.number="editing.maxTokens"
            class="field__input"
            type="number"
            min="1"
            placeholder="Anthropic 默认 16000"
          />
        </label>

        <label class="field">
          <span class="field__label">token 参数名</span>
          <select v-model="editing.maxTokensParam" class="field__input">
            <option value="max_tokens">max_tokens（兼容面最广）</option>
            <option value="max_completion_tokens">max_completion_tokens（OpenAI 新写法）</option>
          </select>
        </label>

        <label class="field">
          <span class="field__label">temperature</span>
          <input
            v-model.number="editing.temperature"
            class="field__input"
            type="number"
            step="0.1"
            placeholder="留空 = 不发送"
          />
          <span class="field__hint">
            Anthropic 已移除采样参数，这一项对它无效
          </span>
        </label>

        <label class="field">
          <span class="field__label">超时（毫秒）</span>
          <input
            v-model.number="editing.timeoutMs"
            class="field__input"
            type="number"
            min="1000"
            step="1000"
            placeholder="120000"
          />
        </label>
      </div>

      <div class="providers__form-actions">
        <button class="btn" :disabled="busy" @click="save">保存</button>
        <button class="btn btn--ghost" :disabled="busy" @click="loadModels">
          拉取模型列表
        </button>
        <button class="btn btn--ghost" @click="cancel">取消</button>
      </div>

      <div v-if="modelOptions.length" class="providers__models">
        <button
          v-for="m in modelOptions"
          :key="m"
          class="providers__model"
          @click="editing.model = m"
        >
          {{ m }}
        </button>
      </div>
    </div>
  </section>
</template>
