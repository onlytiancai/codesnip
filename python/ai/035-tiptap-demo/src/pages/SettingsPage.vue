<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import ProviderPanel from '../components/ProviderPanel.vue'
import PromptPanel from '../components/PromptPanel.vue'
import DictionaryPanel from '../components/DictionaryPanel.vue'
import AccountPanel from '../components/AccountPanel.vue'
import PlanPanel from '../components/PlanPanel.vue'
import AdminPanel from '../components/AdminPanel.vue'
import { useAuth } from '../composables/useAuth'
import { fetchConfig, type ProviderConfig } from '../services/settings'

/**
 * 设置页：LLM provider + 每个 provider 各自的提示词 + 词库 + 账号 + 套餐 / 用户管理。
 *
 * 和编辑器是两个独立路由（/ 和 /settings），不是弹层 ——
 * 提示词编辑要占的地方不小，弹层里塞不下。
 * 侧栏在两个页面都在，所以「← 回到编辑器」之外也能直接切翻译方向。
 *
 * 「用户管理」tab 只有管理员能看见（后端也会再挡一次）。
 *
 * 当前 tab 记在 URL 的 `?tab=` 上 —— 侧栏的「修改密码」能直接跳到某个 tab，
 * 刷新页面也不会跳回第一个。
 */

type Tab = 'providers' | 'prompts' | 'dictionary' | 'account' | 'plan' | 'admin'

const TABS: Tab[] = ['providers', 'prompts', 'dictionary', 'account', 'plan', 'admin']

const route = useRoute()
const router = useRouter()

const tab = ref<Tab>('providers')
const providers = ref<ProviderConfig[]>([])
const activeProviderId = ref('')
const error = ref('')
const loading = ref(true)

const { user } = useAuth()
const isAdmin = computed(() => user.value?.role === 'admin')

function isTab(v: unknown): v is Tab {
  return typeof v === 'string' && (TABS as string[]).includes(v)
}

/** 切 tab：同步到 URL（?tab=account 之类）。 */
function selectTab(next: Tab) {
  tab.value = next
  void router.replace({ query: next === 'providers' ? {} : { tab: next } })
}

/** URL 里的 tab 变了（侧栏跳过来 / 手动改地址栏）也跟着切。 */
watch(
  () => route.query.tab,
  (v) => {
    if (!isTab(v)) return
    if (v === 'admin' && !isAdmin.value) {
      selectTab('providers')
      return
    }
    tab.value = v
  },
)

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

onMounted(() => {
  // 首屏按 URL 里的 tab 来（侧栏「修改密码」会带 ?tab=account）
  const q = route.query.tab
  if (isTab(q) && !(q === 'admin' && !isAdmin.value)) {
    tab.value = q
  }
  void load()
})
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
        @click="selectTab('dictionary')"
      >
        词库
      </button>
      <button
        class="settings__tab"
        :class="{ 'settings__tab--on': tab === 'account' }"
        @click="selectTab('account')"
      >
        账号
      </button>
      <button
        class="settings__tab"
        :class="{ 'settings__tab--on': tab === 'plan' }"
        @click="selectTab('plan')"
      >
        我的套餐
      </button>
      <button
        v-if="isAdmin"
        class="settings__tab"
        :class="{ 'settings__tab--on': tab === 'admin' }"
        @click="selectTab('admin')"
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
      <AccountPanel v-else-if="tab === 'account'" />
      <PlanPanel v-else-if="tab === 'plan'" />
      <AdminPanel v-else-if="tab === 'admin'" />
    </main>
  </div>
</template>
