<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import {
  createCustomPromptFor,
  deleteCustomPrompt,
  fetchCustomPrompts,
  updateCustomPrompt,
  type UserPrompt,
} from '../services/customPrompts'
import {
  MAX_TOOLBAR_PROMPTS,
  PROMPT_ICONS,
  SCOPE_OPTIONS,
  SCOPE_LABEL,
  type PromptScope,
} from '../constants/customPrompts'
import { canCustomizePrompts } from '../composables/useCustomPrompts'
import { ApiError } from '../services/http'
import type { ProviderConfig } from '../services/settings'

/**
 * 「我的指令」—— 用户自己写的提示词。
 *
 * 和「系统提示词」的区别：
 *  - 系统提示词是管理员维护的全局配置（翻译 / 选区 / 整句 / 语法 / 指令）
 *  - 我的指令是用户自己的，**每个 provider 一套** —— 换供应商就是另一套
 *
 * 指令只是一段自然语言（比如「改成学术腔」），
 * 跑起来时由系统提示词（rewriteInstruction 能力）套住它。
 * 「作用范围」决定改哪一段：当前句 / 当前段 / 所有正文。
 */

const props = defineProps<{
  /** 当前生效的 provider（默认选中它） */
  activeProviderId: string
  providers?: ProviderConfig[]
}>()

const providerId = ref(props.activeProviderId)
const prompts = ref<UserPrompt[]>([])
const busy = ref(false)
const error = ref('')
const message = ref('')

/** 正在编辑的那条；'new' = 新建；null = 没在编辑 */
const editing = ref<UserPrompt | 'new' | null>(null)
const form = ref({
  label: '',
  icon: '✨',
  scope: 'sentence' as PromptScope,
  instruction: '',
  showInToolbar: true,
})

const canEdit = computed(() => canCustomizePrompts())
const pinnedCount = computed(() => prompts.value.filter((p) => p.showInToolbar).length)

watch(
  () => props.activeProviderId,
  (id) => {
    if (id && id !== providerId.value) providerId.value = id
  },
)

watch(providerId, (id) => {
  void load(id)
})

onMounted(() => {
  void load(providerId.value)
})

async function load(id: string) {
  if (!id) return
  busy.value = true
  error.value = ''
  try {
    const res = await fetchCustomPrompts(id)
    prompts.value = res.prompts
  } catch (err) {
    error.value =
      err instanceof ApiError ? `[${err.status}] ${err.message}` : String(err)
    prompts.value = []
  } finally {
    busy.value = false
  }
}

// ---- 表单 ----

function startCreate() {
  editing.value = 'new'
  form.value = {
    label: '',
    icon: '✨',
    scope: 'sentence',
    instruction: '',
    showInToolbar: true,
  }
  error.value = ''
  message.value = ''
}

function startEdit(p: UserPrompt) {
  editing.value = p
  form.value = {
    label: p.label,
    icon: p.icon,
    scope: p.scope,
    instruction: p.instruction,
    showInToolbar: p.showInToolbar,
  }
  error.value = ''
  message.value = ''
}

function cancelEdit() {
  editing.value = null
}

async function save() {
  const label = form.value.label.trim()
  const instruction = form.value.instruction.trim()
  if (!label) {
    error.value = '给它起个名字'
    return
  }
  if (!instruction) {
    error.value = '写点指令内容 —— 比如「改成学术腔」'
    return
  }
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    if (editing.value === 'new') {
      await createCustomPromptFor(providerId.value, {
        label,
        icon: form.value.icon,
        scope: form.value.scope,
        instruction,
        showInToolbar: form.value.showInToolbar,
      })
      message.value = '已新建'
    } else if (editing.value) {
      await updateCustomPrompt(editing.value.id, {
        label,
        icon: form.value.icon,
        scope: form.value.scope,
        instruction,
        showInToolbar: form.value.showInToolbar,
      })
      message.value = '已保存'
    }
    editing.value = null
    await load(providerId.value)
  } catch (err) {
    error.value =
      err instanceof ApiError ? `[${err.status}] ${err.message}` : String(err)
  } finally {
    busy.value = false
  }
}

async function remove(p: UserPrompt) {
  if (!window.confirm(`删掉指令「${p.label}」？`)) return
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    await deleteCustomPrompt(p.id)
    message.value = '已删除'
    await load(providerId.value)
  } catch (err) {
    error.value =
      err instanceof ApiError ? `[${err.status}] ${err.message}` : String(err)
  } finally {
    busy.value = false
  }
}

/** 勾「显示在工具栏」—— 最多 5 个。 */
async function toggleToolbar(p: UserPrompt) {
  if (!p.showInToolbar && pinnedCount.value >= MAX_TOOLBAR_PROMPTS) {
    error.value = `工具栏最多放 ${MAX_TOOLBAR_PROMPTS} 个指令 —— 先取消一个再勾这个`
    return
  }
  busy.value = true
  error.value = ''
  try {
    await updateCustomPrompt(p.id, {
      label: p.label,
      icon: p.icon,
      scope: p.scope,
      instruction: p.instruction,
      showInToolbar: !p.showInToolbar,
    })
    await load(providerId.value)
  } catch (err) {
    error.value =
      err instanceof ApiError ? `[${err.status}] ${err.message}` : String(err)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <section class="user-prompts">
    <div class="prompts__bar">
      <label class="field field--inline">
        <span class="field__label">Provider</span>
        <select v-model="providerId" class="field__input">
          <option v-if="!providers?.length" :value="providerId">
            {{ providerId || '（没有生效的 provider）' }}
          </option>
          <option v-for="p in providers ?? []" :key="p.id" :value="p.id">
            {{ p.label }}（{{ p.id }}）{{ p.id === activeProviderId ? ' · 使用中' : '' }}
          </option>
        </select>
      </label>
      <span class="prompts__hint">
        指令跟着供应商走 —— 每个供应商下是你自己那套，互不影响。
      </span>
    </div>

    <!-- 免费版：只能看，不能加 -->
    <div v-if="!canEdit" class="user-prompts__locked">
      <p>
        <strong>自定义指令是付费功能。</strong>
        升级套餐后就能写自己的指令，钉在英文编辑器的工具栏上。
      </p>
      <RouterLink to="/settings?tab=plan" class="btn btn--ghost">看看套餐</RouterLink>
    </div>

    <template v-else>
      <div class="prompts__actions">
        <button class="btn" :disabled="busy" @click="startCreate">+ 新建指令</button>
        <span class="prompts__hint">
          指令只是一句话（比如「改成学术腔」）—— 作用范围决定它改哪一段：
          当前句 / 当前段 / 所有正文。
        </span>
      </div>

      <p v-if="error" class="prompts__error">{{ error }}</p>
      <p v-if="message" class="prompts__ok">{{ message }}</p>

      <!-- 列表 -->
      <ul class="user-prompts__list">
        <li v-for="p in prompts" :key="p.id" class="user-prompts__item">
          <div class="user-prompts__head">
            <span class="user-prompts__icon">{{ p.icon }}</span>
            <span class="user-prompts__label">{{ p.label }}</span>
            <span class="user-prompts__scope">{{ SCOPE_LABEL[p.scope] }}</span>
            <label class="user-prompts__pin" :title="`最多 ${MAX_TOOLBAR_PROMPTS} 个`">
              <input
                type="checkbox"
                :checked="p.showInToolbar"
                :disabled="busy"
                @change="toggleToolbar(p)"
              />
              显示在工具栏
            </label>
            <span class="user-prompts__spacer" />
            <button class="btn btn--ghost btn--sm" @click="startEdit(p)">编辑</button>
            <button class="btn btn--ghost btn--sm" :disabled="busy" @click="remove(p)">
              删除
            </button>
          </div>
          <p class="user-prompts__body">{{ p.instruction }}</p>
        </li>
      </ul>

      <p v-if="!prompts.length && !busy" class="prompts__desc">
        还没有指令 —— 点上面「新建指令」写一个。
      </p>

      <!-- 编辑 / 新建表单 -->
      <form v-if="editing" class="user-prompts__form" @submit.prevent="save">
        <h3 class="prompts__title">
          {{ editing === 'new' ? '新建指令' : `编辑「${editing.label}」` }}
        </h3>

        <label class="field">
          <span class="field__label">名字（工具栏上显示的）</span>
          <input
            v-model="form.label"
            class="field__input"
            placeholder="比如：学术化 / 缩短 / 加例子"
            maxlength="40"
          />
        </label>

        <label class="field">
          <span class="field__label">图标</span>
          <div class="user-prompts__icons">
            <button
              v-for="ic in PROMPT_ICONS"
              :key="ic"
              type="button"
              class="user-prompts__icon-btn"
              :class="{ 'is-on': form.icon === ic }"
              @click="form.icon = ic"
            >
              {{ ic }}
            </button>
          </div>
        </label>

        <div class="field">
          <span class="field__label">作用范围</span>
          <label v-for="opt in SCOPE_OPTIONS" :key="opt.value" class="prompts__mode">
            <input v-model="form.scope" type="radio" :value="opt.value" />
            {{ opt.label }}
            <span class="field__hint">{{ opt.hint }}</span>
          </label>
        </div>

        <label class="field">
          <span class="field__label">指令内容</span>
          <textarea
            v-model="form.instruction"
            class="field__input field__textarea"
            rows="3"
            placeholder="比如：改写成学术论文的语气，保持原意，长度不变。"
            maxlength="2000"
          />
        </label>

        <label class="prompts__mode">
          <input v-model="form.showInToolbar" type="checkbox" />
          显示在英文编辑器的工具栏上
        </label>

        <div class="prompts__actions">
          <button class="btn" type="submit" :disabled="busy">保存</button>
          <button class="btn btn--ghost" type="button" @click="cancelEdit">取消</button>
        </div>
      </form>
    </template>
  </section>
</template>
