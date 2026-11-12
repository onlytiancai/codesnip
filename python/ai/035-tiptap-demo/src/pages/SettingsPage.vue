<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import ProviderPanel from '../components/ProviderPanel.vue'
import PromptPanel from '../components/PromptPanel.vue'
import DictionaryPanel from '../components/DictionaryPanel.vue'
import PlanPanel from '../components/PlanPanel.vue'
import AdminPanel from '../components/AdminPanel.vue'
import { useAuth } from '../composables/useAuth'
import { fetchConfig, type ProviderConfig } from '../services/settings'

/**
 * 设置页：LLM provider + 每个 provider 各自的提示词 + 词库 + 套餐 / 用户管理。
 *
 * 和编辑器是两个独立路由（/ 和 /settings），不是弹层 ——
 * 提示词编辑要占的地方不小，弹层里塞不下。
 * 侧栏在两个页面都在，所以「← 回到编辑器」之外也能直接切翻译方向。
 *
 * 「用户管理」tab 只有管理员能看见（后端也会再挡一次）。
 */

type Tab = 'providers' | 'prompts' | 'dictionary' | 'plan' | 'admin'

const tab = ref<Tab>('providers')
const providers = ref<ProviderConfig[]>([])
const activeProviderId = ref('')
const error = ref('')
const loading = ref(true)

const { user } = useAuth()
const isAdmin = computed(() => user.value?.role === 'admin')

async function load() {
  loading.value = true
  error.value = ''
  try {
    const cfg = await fetchConfig()
    providers.value = cfg.providers
    activeProviderId.value = cfg.activeProviderId
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="settings">
    <header class="settings__header">
      <div class="settings__title-wrap">
        <h1 class="settings__title">设置</h1>
        <span v-if="activeProviderId" class="settings__active">
          当前生效：{{ providers.find((p) => p.id === activeProviderId)?.label ?? activeProviderId }}
        </span>
      </div>
      <RouterLink to="/" class="btn btn--ghost">← 回到编辑器</RouterLink>
    </header>

    <nav class="settings__tabs">
      <button
        class="settings__tab"
        :class="{ 'settings__tab--on': tab === 'providers' }"
        @click="tab = 'providers'"
      >
        LLM 供应商
      </button>
      <button
        class="settings__tab"
        :class="{ 'settings__tab--on': tab === 'prompts' }"
        @click="tab = 'prompts'"
      >
        提示词
      </button>
      <button
        class="settings__tab"
        :class="{ 'settings__tab--on': tab === 'dictionary' }"
        @click="tab = 'dictionary'"
      >
        词库
      </button>
      <button
        class="settings__tab"
        :class="{ 'settings__tab--on': tab === 'plan' }"
        @click="tab = 'plan'"
      >
        我的套餐
      </button>
      <button
        v-if="isAdmin"
        class="settings__tab"
        :class="{ 'settings__tab--on': tab === 'admin' }"
        @click="tab = 'admin'"
      >
        用户管理
      </button>
    </nav>

    <p v-if="error" class="settings__error">{{ error }}</p>
    <p v-if="loading" class="settings__loading">加载中…</p>

    <main v-else class="settings__main">
      <ProviderPanel
        v-if="tab === 'providers'"
        :providers="providers"
        :active-provider-id="activeProviderId"
        @changed="load"
      />
      <PromptPanel
        v-else-if="tab === 'prompts'"
        :providers="providers"
        :active-provider-id="activeProviderId"
      />
      <DictionaryPanel v-else-if="tab === 'dictionary'" />
      <PlanPanel v-else-if="tab === 'plan'" />
      <AdminPanel v-else-if="tab === 'admin'" />
    </main>
  </div>
</template>
