<script setup lang="ts">
import { onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { doLogin, doRegister } from '../composables/useAuth'
import {
  resetPassword,
  sendRegisterCode,
  sendResetCode,
} from '../services/auth'
import { debugLog } from '../utils/debug'

/**
 * 登录页。
 *
 * 独立的顶层路由（不套 AppSidebar 外壳）—— 登录前没有侧栏可言。
 * 三种模式：
 *   - 登录（用户名或邮箱 + 密码）
 *   - 注册（用户名 + 邮箱 + 邮箱验证码 + 密码）
 *   - 忘记密码（邮箱 + 邮箱验证码 + 新密码）
 */

type Mode = 'login' | 'register' | 'reset'

const router = useRouter()
const route = useRoute()

const mode = ref<Mode>('login')
const identifier = ref('')
const username = ref('')
const email = ref('')
const password = ref('')
const code = ref('')
const newPassword = ref('')
const error = ref('')
const info = ref('')
const busy = ref(false)

// 从「账号 → 用邮箱验证码重置」跳回来的：提示一下为什么被踢下线
if (route.query.reset === '1') {
  info.value = '密码已重置，请用新密码登录'
}

/** 发了码之后的冷却倒计时（秒）—— 期间按钮禁用。 */
const cooldown = ref(0)
let timer: number | undefined

onUnmounted(() => {
  if (timer !== undefined) window.clearInterval(timer)
})

function startCooldown(seconds: number) {
  cooldown.value = seconds
  if (timer !== undefined) window.clearInterval(timer)
  timer = window.setInterval(() => {
    cooldown.value -= 1
    if (cooldown.value <= 0 && timer !== undefined) {
      window.clearInterval(timer)
      timer = undefined
    }
  }, 1000)
}

/** 发验证码（注册 / 重置密码共用）。 */
async function sendCode() {
  const mail = email.value.trim()
  if (!mail) {
    error.value = '先填邮箱'
    return
  }
  error.value = ''
  busy.value = true
  try {
    const res =
      mode.value === 'register'
        ? await sendRegisterCode(mail)
        : await sendResetCode(mail)
    startCooldown(res.cooldown || 60)
    info.value = `验证码已发到 ${mail}，10 分钟内有效`
    debugLog('[auth] 验证码已发送:', mail)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

async function submit() {
  error.value = ''
  info.value = ''
  busy.value = true
  try {
    if (mode.value === 'login') {
      if (!identifier.value.trim() || !password.value) {
        error.value = '用户名/邮箱和密码都要填'
        return
      }
      await doLogin(identifier.value.trim(), password.value)
      debugLog('[auth] 登录页提交成功，跳回编辑器')
      await router.push('/')
    } else if (mode.value === 'register') {
      if (
        !username.value.trim() ||
        !email.value.trim() ||
        !code.value.trim() ||
        !password.value
      ) {
        error.value = '用户名、邮箱、验证码、密码都要填'
        return
      }
      await doRegister(
        username.value.trim(),
        email.value.trim(),
        password.value,
        code.value.trim(),
      )
      debugLog('[auth] 注册页提交成功，跳回编辑器')
      await router.push('/')
    } else {
      if (!email.value.trim() || !code.value.trim() || !newPassword.value) {
        error.value = '邮箱、验证码、新密码都要填'
        return
      }
      await resetPassword(
        email.value.trim(),
        code.value.trim(),
        newPassword.value,
      )
      info.value = '密码已重置，用新密码去登录吧'
      debugLog('[auth] 密码已重置')
      mode.value = 'login'
      password.value = ''
      code.value = ''
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    debugLog('[auth] 登录页提交失败:', error.value)
  } finally {
    busy.value = false
  }
}

function switchMode(next: Mode) {
  mode.value = next
  error.value = ''
  info.value = ''
  code.value = ''
  // 各模式只清自己用不到的输入框
  if (next === 'reset') {
    password.value = ''
  } else if (next === 'register') {
    newPassword.value = ''
  }
}

const TITLE: Record<Mode, string> = {
  login: '登录',
  register: '注册',
  reset: '重置密码',
}
</script>

<template>
  <div class="login">
    <form class="login__card" @submit.prevent="submit">
      <div class="login__brand">
        <span class="login__logo">🌐</span>
        <h1 class="login__brand-title">翻译工作台</h1>
      </div>
      <p class="login__subtitle">中英双向逐句翻译 · 写作 AI 助手</p>

      <hr class="login__divider" />

      <h2 class="login__title">{{ TITLE[mode] }}</h2>
      <p class="login__hint">
        {{
          mode === 'login'
            ? '登录后才能使用翻译和 AI 改写功能。'
            : mode === 'register'
              ? '注册要填邮箱 —— 验证码会发到这个邮箱。'
              : '忘了密码？用邮箱验证码重置，不用记当前密码。'
        }}
      </p>

      <p v-if="error" class="settings__error">{{ error }}</p>
      <p v-if="info" class="plans__ok">{{ info }}</p>

      <!-- 登录 -->
      <template v-if="mode === 'login'">
        <label class="field">
          <span class="field__label">用户名或邮箱</span>
          <input
            v-model="identifier"
            class="field__input"
            autocomplete="username"
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
      </template>

      <!-- 注册 -->
      <template v-else-if="mode === 'register'">
        <label class="field">
          <span class="field__label">用户名</span>
          <input
            v-model="username"
            class="field__input"
            autocomplete="username"
            @keyup.enter="submit"
          />
        </label>

        <label class="field">
          <span class="field__label">邮箱</span>
          <input
            v-model="email"
            class="field__input"
            type="email"
            autocomplete="email"
            @keyup.enter="sendCode"
          />
        </label>

        <label class="field">
          <span class="field__label">邮箱验证码</span>
          <div class="login__code-row">
            <input
              v-model="code"
              class="field__input"
              inputmode="numeric"
              maxlength="6"
              autocomplete="one-time-code"
              @keyup.enter="submit"
            />
            <button
              type="button"
              class="btn"
              :disabled="busy || cooldown > 0"
              @click="sendCode"
            >
              {{ cooldown > 0 ? `${cooldown}s` : '发送验证码' }}
            </button>
          </div>
        </label>

        <label class="field">
          <span class="field__label">密码</span>
          <input
            v-model="password"
            class="field__input"
            type="password"
            autocomplete="new-password"
            @keyup.enter="submit"
          />
        </label>
      </template>

      <!-- 重置密码 -->
      <template v-else>
        <label class="field">
          <span class="field__label">邮箱</span>
          <input
            v-model="email"
            class="field__input"
            type="email"
            autocomplete="email"
            @keyup.enter="sendCode"
          />
        </label>

        <label class="field">
          <span class="field__label">邮箱验证码</span>
          <div class="login__code-row">
            <input
              v-model="code"
              class="field__input"
              inputmode="numeric"
              maxlength="6"
              autocomplete="one-time-code"
              @keyup.enter="submit"
            />
            <button
              type="button"
              class="btn"
              :disabled="busy || cooldown > 0"
              @click="sendCode"
            >
              {{ cooldown > 0 ? `${cooldown}s` : '发送验证码' }}
            </button>
          </div>
        </label>

        <label class="field">
          <span class="field__label">新密码</span>
          <input
            v-model="newPassword"
            class="field__input"
            type="password"
            autocomplete="new-password"
            @keyup.enter="submit"
          />
        </label>
      </template>

      <button class="btn login__submit" type="submit" :disabled="busy">
        {{ busy ? '处理中…' : TITLE[mode] }}
      </button>

      <p class="login__switch">
        <template v-if="mode === 'login'">
          没有账号？
          <button type="button" class="login__link" @click="switchMode('register')">
            注册一个
          </button>
          ·
          <button type="button" class="login__link" @click="switchMode('reset')">
            忘记密码
          </button>
        </template>
        <template v-else-if="mode === 'register'">
          已有账号？
          <button type="button" class="login__link" @click="switchMode('login')">
            去登录
          </button>
        </template>
        <template v-else>
          想起密码了？
          <button type="button" class="login__link" @click="switchMode('login')">
            去登录
          </button>
        </template>
      </p>
    </form>
  </div>
</template>
