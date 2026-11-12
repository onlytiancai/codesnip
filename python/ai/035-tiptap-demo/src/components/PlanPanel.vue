<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { refreshUsage, useAuth } from '../composables/useAuth'
import {
  fetchMyPlanRequests,
  fetchPlans,
  redeemInvite,
  requestPlanUpgrade,
  type PlanRequestInfo,
} from '../services/plans'
import type { PlanInfo } from '../services/auth'
import { debugLog } from '../utils/debug'

/**
 * 「我的套餐」面板（设置页的一个 tab）。
 *
 * 三块：
 *   1. 当前套餐 + 今日/本月配额条（token，输入+输出）
 *   2. 升级：邀请码兑换（立刻生效）或者提交申请（管理员审核后生效）
 *   3. 我的申请记录
 *
 * 配额超了之后 AI 接口会返回 403 —— 面板上的进度条会变红提示。
 */

const { user } = useAuth()

const plans = ref<PlanInfo[]>([])
const requests = ref<PlanRequestInfo[]>([])
const inviteCode = ref('')
const targetPlan = ref('')
const note = ref('')
const error = ref('')
const message = ref('')
const busy = ref(false)

async function load() {
  error.value = ''
  try {
    const [p, r] = await Promise.all([fetchPlans(), fetchMyPlanRequests()])
    plans.value = p.plans
    requests.value = r.requests
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

onMounted(load)

function fmt(n: number | undefined): string {
  return (n ?? 0).toLocaleString('zh-CN')
}

/** 配额条百分比（0~100）。上限 0 = 不限 → 返回 0。 */
function percent(used: number, limit: number): number {
  if (limit <= 0) return 0
  return Math.min(100, Math.round((used / limit) * 100))
}

/** 除了当前套餐，还能申请哪些（比当前档高的、且允许申请的）。 */
function upgradeablePlans(): PlanInfo[] {
  const cur = user.value?.plan
  if (!cur) return []
  return plans.value.filter(
    (p) =>
      p.id !== cur.id &&
      p.upgradeable &&
      (p.dailyLimit > cur.dailyLimit || p.monthlyLimit > cur.monthlyLimit),
  )
}

/** 兑换邀请码 —— 立刻升级。 */
async function redeem() {
  const code = inviteCode.value.trim()
  if (!code) {
    error.value = '请输入邀请码'
    return
  }
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    const res = await redeemInvite(code)
    inviteCode.value = ''
    message.value = `兑换成功，已升级到「${res.plan.label}」`
    debugLog('[plan] 邀请码兑换成功:', res.plan.id)
    await refreshUsage()
    await load()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

/** 提交升级申请 —— 等管理员审核。 */
async function submitRequest() {
  if (!targetPlan.value) {
    error.value = '请选择要升级的套餐'
    return
  }
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    await requestPlanUpgrade(targetPlan.value, note.value.trim() || undefined)
    targetPlan.value = ''
    note.value = ''
    message.value = '申请已提交，等管理员审核'
    debugLog('[plan] 提交升级申请:', targetPlan.value)
    await load()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

const STATUS_LABEL: Record<PlanRequestInfo['status'], string> = {
  pending: '待审核',
  approved: '已通过',
  rejected: '已驳回',
}
</script>

<template>
  <section class="plans">
    <p v-if="error" class="plans__error">{{ error }}</p>
    <p v-if="message" class="plans__ok">{{ message }}</p>

    <!-- 当前套餐 + 配额 -->
    <div v-if="user" class="plans__current">
      <div class="plans__current-head">
        <strong class="plans__current-name">{{ user.plan.label }}</strong>
        <span class="plans__current-desc">{{ user.plan.description }}</span>
      </div>

      <div class="plans__quota">
        <div class="plans__quota-row">
          <span class="plans__quota-tag">今日</span>
          <div class="plans__bar">
            <div
              class="plans__bar-fill"
              :class="{
                'plans__bar-fill--over': user.quota.dailyExceeded,
              }"
              :style="{
                width: `${percent(user.quota.dailyUsed, user.quota.dailyLimit)}%`,
              }"
            />
          </div>
          <span class="plans__quota-num">
            {{ fmt(user.quota.dailyUsed) }} /
            {{ user.quota.dailyLimit > 0 ? fmt(user.quota.dailyLimit) : '不限' }}
          </span>
        </div>

        <div class="plans__quota-row">
          <span class="plans__quota-tag">本月</span>
          <div class="plans__bar">
            <div
              class="plans__bar-fill"
              :class="{
                'plans__bar-fill--over': user.quota.monthlyExceeded,
              }"
              :style="{
                width: `${percent(user.quota.monthlyUsed, user.quota.monthlyLimit)}%`,
              }"
            />
          </div>
          <span class="plans__quota-num">
            {{ fmt(user.quota.monthlyUsed) }} /
            {{ user.quota.monthlyLimit > 0 ? fmt(user.quota.monthlyLimit) : '不限' }}
          </span>
        </div>
      </div>

      <p v-if="user.quota.dailyExceeded" class="plans__warn">
        今日配额已用完 —— AI 功能暂时用不了，升级套餐或等明天自动恢复。
      </p>
      <p v-else-if="user.quota.monthlyExceeded" class="plans__warn">
        本月配额已用完 —— AI 功能暂时用不了，升级套餐或等下个月自动恢复。
      </p>
    </div>

    <!-- 升级：邀请码 / 申请 -->
    <div class="plans__upgrade">
      <h3 class="plans__title">升级套餐</h3>
      <p class="plans__hint">
        两条路：① 用<b>邀请码</b>兑换（管理员发的，立刻生效）；②
        <b>提交申请</b>，管理员审核通过后生效。
      </p>

      <div class="plans__upgrade-grid">
        <div class="plans__block">
          <span class="field__label">邀请码</span>
          <div class="plans__row">
            <input
              v-model="inviteCode"
              class="field__input"
              placeholder="XXXX-XXXX-XXXX"
              @keyup.enter="redeem"
            />
            <button class="btn" :disabled="busy" @click="redeem">兑换</button>
          </div>
        </div>

        <div class="plans__block">
          <span class="field__label">申请升级</span>
          <div class="plans__row">
            <select v-model="targetPlan" class="field__input">
              <option value="">选择套餐…</option>
              <option v-for="p in upgradeablePlans()" :key="p.id" :value="p.id">
                {{ p.label }} —— {{ p.description }}
              </option>
            </select>
            <button class="btn" :disabled="busy || !targetPlan" @click="submitRequest">
              提交申请
            </button>
          </div>
          <textarea
            v-model="note"
            class="field__input plans__note"
            rows="2"
            placeholder="备注（可选）：为什么需要升级"
          ></textarea>
        </div>
      </div>
    </div>

    <!-- 申请记录 -->
    <div class="plans__history">
      <h3 class="plans__title">我的申请</h3>
      <p v-if="requests.length === 0" class="plans__hint">还没有提交过升级申请。</p>
      <ul v-else class="plans__req-list">
        <li v-for="r in requests" :key="r.id" class="plans__req">
          <span class="plans__req-plan">{{ r.toPlanLabel }}</span>
          <span class="plans__req-status" :class="`plans__req-status--${r.status}`">
            {{ STATUS_LABEL[r.status] }}
          </span>
          <span class="plans__req-time">{{ r.createdAt.slice(0, 10) }}</span>
          <span v-if="r.reviewNote" class="plans__req-note">审核意见：{{ r.reviewNote }}</span>
        </li>
      </ul>
    </div>
  </section>
</template>
