<script setup lang="ts">
import { onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { doLogout, useAuth } from '../composables/useAuth'
import {
  changePassword,
  resetPassword,
  sendResetCode,
} from '../services/auth'
import { debugLog } from '../utils/debug'

/**
 * 「账号」面板（设置页的一个 tab）。
 *
 * 三块：
 *   1. 账号信息（用户名 / 邮箱 / 角色 / 套餐）
 *   2. 修改密码 —— 两个入口：
 *      · 用当前密码改（记得旧密码时用）
 *      · 用邮箱验证码重置（忘了旧密码时用）
 *
 * 密码规则跟着后端走：至少 4 位。后端也会再校验一次，前端只是提前拦一道。
 */

const MIN_PASSWORD_LENGTH = 4

const { user } = useAuth()
const router = useRouter()

type Method = 'current' | 'email'
const method = ref<Method>('current')

// —— 方式一：当前密码 ——
const currentPassword = ref('')
const newPassword = ref('')
const confirmPassword = ref('')

// —— 方式二：邮箱验证码 ——
const resetCode = ref('')
const resetNewPassword = ref('')

const error = ref('')
const message = ref('')
const busy = ref(false)

/** 发了码之后的冷却倒计时（秒）。 */
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

/** 发重置密码的验证码。 */
async function sendCode() {
  const mail = user.value?.email
  if (!mail) {
    error.value = '这个账号还没绑定邮箱，请联系管理员补一个'
    return
  }
  error.value = ''
  message.value = ''
  busy.value = true
  try {
    const res = await sendResetCode(mail)
    startCooldown(res.cooldown || 60)
    message.value = `验证码已发到 ${mail}，10 分钟内有效`
    debugLog('[auth] 重置密码验证码已发送:', mail)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

/** 用当前密码改。 */
async function submitCurrent() {
  error.value = ''
  message.value = ''

  if (!currentPassword.value) {
    error.value = '请输入当前密码'
    return
  }
  if (newPassword.value.length < MIN_PASSWORD_LENGTH) {
    error.value = `新密码至少 ${MIN_PASSWORD_LENGTH} 位`
    return
  }
  if (newPassword.value !== confirmPassword.value) {
    error.value = '两次输入的新密码不一样'
    return
  }
  if (newPassword.value === currentPassword.value) {
    error.value = '新密码不能和当前密码一样'
    return
  }

  busy.value = true
  try {
    await changePassword(currentPassword.value, newPassword.value)
    currentPassword.value = ''
    newPassword.value = ''
    confirmPassword.value = ''
    message.value = '密码已修改，其它设备上的登录已失效'
    debugLog('[auth] 密码修改成功')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    debugLog('[auth] 密码修改失败:', error.value)
  } finally {
    busy.value = false
  }
}

/** 用邮箱验证码重置（不用记当前密码）。 */
async function submitReset() {
  error.value = ''
  message.value = ''

  if (!resetCode.value.trim()) {
    error.value = '请输入邮箱验证码'
    return
  }
  if (resetNewPassword.value.length < MIN_PASSWORD_LENGTH) {
    error.value = `新密码至少 ${MIN_PASSWORD_LENGTH} 位`
    return
  }

  const mail = user.value?.email
  if (!mail) {
    error.value = '这个账号还没绑定邮箱，请联系管理员补一个'
    return
  }

  busy.value = true
  try {
    await resetPassword(mail, resetCode.value.trim(), resetNewPassword.value)
    resetCode.value = ''
    resetNewPassword.value = ''
    message.value = '密码已重置，所有设备都需要重新登录'
    debugLog('[auth] 密码已重置（邮箱验证码）')
    // 后端把**所有**会话都作废了 —— 本地登录态也一起清掉，然后回登录页
    await doLogout()
    await router.push({ name: 'login', query: { reset: '1' } })
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    debugLog('[auth] 密码重置失败:', error.value)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <section class="account">
    <!-- 账号信息 -->
    <div v-if="user" class="account__info">
      <div class="account__info-row">
        <span class="account__info-label">用户名</span>
        <span class="account__info-value">{{ user.username }}</span>
      </div>
      <div class="account__info-row">
        <span class="account__info-label">邮箱</span>
        <span class="account__info-value">
          {{ user.email ?? '未绑定（请联系管理员补一个）' }}
        </span>
      </div>
      <div class="account__info-row">
        <span class="account__info-label">角色</span>
        <span class="account__info-value">
          {{ user.role === 'admin' ? '管理员' : '普通用户' }}
        </span>
      </div>
      <div class="account__info-row">
        <span class="account__info-label">套餐</span>
        <span class="account__info-value">{{ user.plan.label }}</span>
      </div>
    </div>

    <!-- 修改密码 -->
    <div class="account__password">
      <h3 class="plans__title">修改密码</h3>
      <p class="plans__hint">
        两种方式：记得旧密码就用<b>当前密码</b>；忘了就用<b>邮箱验证码</b>重置
        （会把这个账号的所有设备都登出去）。
      </p>

      <div class="account__methods">
        <button
          type="button"
          class="account__method"
          :class="{ 'account__method--on': method === 'current' }"
          @click="method = 'current'"
        >
          用当前密码
        </button>
        <button
          type="button"
          class="account__method"
          :class="{ 'account__method--on': method === 'email' }"
          @click="method = 'email'"
        >
          用邮箱验证码
        </button>
      </div>

      <p v-if="error" class="plans__error">{{ error }}</p>
      <p v-if="message" class="plans__ok">{{ message }}</p>

      <!-- 方式一 -->
      <form v-if="method === 'current'" class="account__form" @submit.prevent="submitCurrent">
        <label class="field">
          <span class="field__label">当前密码</span>
          <input
            v-model="currentPassword"
            class="field__input"
            type="password"
            autocomplete="current-password"
          />
        </label>

        <label class="field">
          <span class="field__label">新密码</span>
          <input
            v-model="newPassword"
            class="field__input"
            type="password"
            autocomplete="new-password"
          />
          <span class="field__hint">至少 {{ MIN_PASSWORD_LENGTH }} 位</span>
        </label>

        <label class="field">
          <span class="field__label">确认新密码</span>
          <input
            v-model="confirmPassword"
            class="field__input"
            type="password"
            autocomplete="new-password"
          />
        </label>

        <button class="btn" type="submit" :disabled="busy">
          {{ busy ? '提交中…' : '修改密码' }}
        </button>
      </form>

      <!-- 方式二 -->
      <form v-else class="account__form" @submit.prevent="submitReset">
        <label class="field">
          <span class="field__label">邮箱验证码</span>
          <div class="login__code-row">
            <input
              v-model="resetCode"
              class="field__input"
              inputmode="numeric"
              maxlength="6"
              autocomplete="one-time-code"
            />
            <button
              type="button"
              class="btn"
              :disabled="busy || cooldown > 0 || !user?.email"
              @click="sendCode"
            >
              {{ cooldown > 0 ? `${cooldown}s` : '发送验证码' }}
            </button>
          </div>
          <span class="field__hint">
            {{ user?.email ? `发到 ${user.email}` : '账号未绑定邮箱' }}
          </span>
        </label>

        <label class="field">
          <span class="field__label">新密码</span>
          <input
            v-model="resetNewPassword"
            class="field__input"
            type="password"
            autocomplete="new-password"
          />
          <span class="field__hint">至少 {{ MIN_PASSWORD_LENGTH }} 位</span>
        </label>

        <button class="btn" type="submit" :disabled="busy">
          {{ busy ? '提交中…' : '重置密码' }}
        </button>
      </form>
    </div>
  </section>
</template>
