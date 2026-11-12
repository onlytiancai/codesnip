<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { doLogin, doRegister } from '../composables/useAuth'
import { debugLog } from '../utils/debug'

/**
 * 登录页。
 *
 * 独立的顶层路由（不套 AppSidebar 外壳）—— 登录前没有侧栏可言。
 * 同一个页面切换「登录 / 注册」两种模式。
 */

const router = useRouter()

const mode = ref<'login' | 'register'>('login')
const username = ref('')
const password = ref('')
const error = ref('')
const busy = ref(false)

async function submit() {
  const name = username.value.trim()
  if (!name || !password.value) {
    error.value = '用户名和密码都要填'
    return
  }
  error.value = ''
  busy.value = true
  try {
    if (mode.value === 'login') {
      await doLogin(name, password.value)
    } else {
      await doRegister(name, password.value)
    }
    debugLog('[auth] 登录页提交成功，跳回编辑器')
    await router.push('/')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    debugLog('[auth] 登录页提交失败:', error.value)
  } finally {
    busy.value = false
  }
}

function switchMode(next: 'login' | 'register') {
  mode.value = next
  error.value = ''
}
</script>

<template>
  <div class="login">
    <form class="login__card" @submit.prevent="submit">
      <h1 class="login__title">{{ mode === 'login' ? '登录' : '注册' }}</h1>
      <p class="login__hint">
        {{
          mode === 'login'
            ? '登录后才能使用翻译和 AI 改写功能。'
            : '注册一个账号 —— 用户名 3-32 位字母/数字/下划线/连字符。'
        }}
      </p>

      <label class="field">
        <span class="field__label">用户名</span>
        <input
          v-model="username"
          class="field__input"
          autocomplete="username"
          placeholder="admin"
          @keyup.enter="submit"
        />
      </label>

      <label class="field">
        <span class="field__label">密码</span>
        <input
          v-model="password"
          class="field__input"
          type="password"
          autocomplete="current-password"
          @keyup.enter="submit"
        />
      </label>

      <p v-if="error" class="settings__error">{{ error }}</p>

      <button class="btn login__submit" type="submit" :disabled="busy">
        {{ busy ? '处理中…' : mode === 'login' ? '登录' : '注册并登录' }}
      </button>

      <p class="login__switch">
        <template v-if="mode === 'login'">
          没有账号？
          <button type="button" class="login__link" @click="switchMode('register')">
            注册一个
          </button>
        </template>
        <template v-else>
          已有账号？
          <button type="button" class="login__link" @click="switchMode('login')">
            去登录
          </button>
        </template>
      </p>
    </form>
  </div>
</template>
