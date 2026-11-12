<script setup lang="ts">
import { ref } from 'vue'
import { useAuth } from '../composables/useAuth'
import { changePassword } from '../services/auth'
import { debugLog } from '../utils/debug'

/**
 * 「账号」面板（设置页的一个 tab）。
 *
 * 两块：
 *   1. 当前账号信息（用户名 / 角色 / 套餐）
 *   2. 修改密码（要填当前密码，改完其它设备上的会话会失效）
 *
 * 密码规则跟着后端走：至少 4 位。后端也会再校验一次，前端只是提前拦一道。
 */

const MIN_PASSWORD_LENGTH = 4

const { user } = useAuth()

const currentPassword = ref('')
const newPassword = ref('')
const confirmPassword = ref('')
const error = ref('')
const message = ref('')
const busy = ref(false)

async function submit() {
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
        要填当前密码确认身份。改完之后，<b>其它设备上的登录会立即失效</b>，
        当前这个浏览器不受影响。
      </p>

      <p v-if="error" class="plans__error">{{ error }}</p>
      <p v-if="message" class="plans__ok">{{ message }}</p>

      <form class="account__form" @submit.prevent="submit">
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
    </div>
  </section>
</template>
