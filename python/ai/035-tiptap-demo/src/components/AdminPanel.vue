<script setup lang="ts">
import { onMounted, ref } from 'vue'
import {
  approvePlanRequest,
  createInviteCodes,
  disableUser,
  enableUser,
  fetchAdminUser,
  fetchAdminUsers,
  fetchInviteCodes,
  fetchPlanRequests,
  rejectPlanRequest,
  revokeInviteCode,
  setUserPlan,
  setUserEmail,
  type AdminUser,
  type AdminUserDetail,
  type InviteCode,
} from '../services/admin'
import type { PlanInfo } from '../services/auth'
import type { PlanRequestInfo } from '../services/plans'
import { debugLog } from '../utils/debug'

/**
 * 管理面板（设置页的「用户管理」tab，只有管理员能看见）。
 *
 * 三块：
 *   1. 用户列表 —— 禁用 / 启用 / 直接改套餐 / 看用量明细
 *   2. 升级审核 —— 通过 / 驳回
 *   3. 邀请码池 —— 批量生成 / 查看 / 作废
 */

const users = ref<AdminUser[]>([])
const requests = ref<PlanRequestInfo[]>([])
const codes = ref<InviteCode[]>([])
const plans = ref<PlanInfo[]>([])

const error = ref('')
const message = ref('')
const busy = ref(false)

/** 展开的用量明细。 */
const detail = ref<AdminUserDetail | null>(null)

/** 每个用户的邮箱输入框草稿（key = 用户 id）。 */
const emailDraft = ref<Record<number, string>>({})

/** 生成邀请码的表单。 */
const genPlanId = ref('')
const genCount = ref(5)
const genBatch = ref('')
/** 生成出来的码（给管理员复制走）。 */
const generated = ref<string[]>([])

async function load() {
  error.value = ''
  try {
    const [u, r, c] = await Promise.all([
      fetchAdminUsers(),
      fetchPlanRequests('pending'),
      fetchInviteCodes(),
    ])
    users.value = u.users
    requests.value = r.requests
    codes.value = c.codes
    plans.value = c.plans
    // 邮箱输入框的初值 = 库里的邮箱（老用户是空）
    const draft: Record<number, string> = {}
    for (const row of u.users) draft[row.id] = row.email ?? ''
    emailDraft.value = draft
    if (!genPlanId.value && c.plans.length) genPlanId.value = c.plans[0].id
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

onMounted(load)

function fmt(n: number | undefined): string {
  return (n ?? 0).toLocaleString('zh-CN')
}

/** 包一层：出错统一提示，成功统一提示 + 重新拉数据。 */
async function run(action: () => Promise<unknown>, okMessage?: string) {
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    await action()
    message.value = okMessage ?? '操作完成'
    await load()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

function toggleDisable(u: AdminUser) {
  void run(
    () => (u.disabled ? enableUser(u.id) : disableUser(u.id)),
    u.disabled ? `已启用 ${u.username}` : `已禁用 ${u.username}`,
  )
}

function changePlan(u: AdminUser, planId: string) {
  if (!planId || planId === u.planId) return
  void run(() => setUserPlan(u.id, planId), `${u.username} 的套餐已改成 ${planId}`)
}

/** 管理员给用户补邮箱。 */
function saveEmail(u: AdminUser) {
  const email = (emailDraft.value[u.id] ?? '').trim()
  if (!email) return
  void run(() => setUserEmail(u.id, email), `${u.username} 的邮箱已设为 ${email}`)
}

function showDetail(u: AdminUser) {
  void (async () => {
    error.value = ''
    try {
      detail.value = await fetchAdminUser(u.id)
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
    }
  })()
}

function review(r: PlanRequestInfo, approve: boolean) {
  void run(
    () => (approve ? approvePlanRequest(r.id) : rejectPlanRequest(r.id)),
    approve ? `已通过 ${r.username} 的升级申请` : `已驳回 ${r.username} 的升级申请`,
  )
}

async function generate() {
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    const res = await createInviteCodes(
      genPlanId.value,
      Number(genCount.value) || 1,
      genBatch.value.trim() || undefined,
    )
    generated.value = res.codes.map((c) => c.code)
    message.value = `已生成 ${res.codes.length} 个 ${res.plan.label} 邀请码`
    debugLog('[admin] 生成邀请码', res.codes.length, '个')
    await load()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    busy.value = false
  }
}

function revoke(code: string) {
  void run(() => revokeInviteCode(code), `已作废 ${code}`)
}

/** 复制刚生成的码（一行一个，方便直接发群里）。 */
async function copyGenerated() {
  try {
    await navigator.clipboard.writeText(generated.value.join('\n'))
    message.value = '已复制到剪贴板'
  } catch {
    error.value = '复制失败 —— 手动选中下面的码复制吧'
  }
}

const STATUS_LABEL: Record<PlanRequestInfo['status'], string> = {
  pending: '待审核',
  approved: '已通过',
  rejected: '已驳回',
}
</script>

<template>
  <section class="admin">
    <p v-if="error" class="admin__error">{{ error }}</p>
    <p v-if="message" class="admin__ok">{{ message }}</p>

    <!-- ============ 用户列表 ============ -->
    <div class="admin__block">
      <h3 class="admin__title">用户（{{ users.length }}）</h3>
      <table class="admin__table">
        <thead>
          <tr>
            <th>用户名</th>
            <th>邮箱</th>
            <th>套餐</th>
            <th>今日 token</th>
            <th>本月 token</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="u in users" :key="u.id">
            <td>
              <strong>{{ u.username }}</strong>
              <span v-if="u.role === 'admin'" class="admin__badge">管理员</span>
            </td>
            <td>
              <div class="admin__email">
                <input
                  v-model="emailDraft[u.id]"
                  class="field__input field__input--sm admin__email-input"
                  placeholder="未绑定"
                />
                <button
                  class="btn btn--ghost btn--sm"
                  :disabled="busy || !emailDraft[u.id] || emailDraft[u.id] === (u.email ?? '')"
                  @click="saveEmail(u)"
                >
                  存
                </button>
              </div>
            </td>
            <td>
              <select
                class="field__input admin__plan-select"
                :value="u.planId"
                :disabled="busy"
                @change="changePlan(u, ($event.target as HTMLSelectElement).value)"
              >
                <option v-for="p in plans" :key="p.id" :value="p.id">{{ p.label }}</option>
              </select>
            </td>
            <td>{{ fmt(u.today.totalTokens) }}</td>
            <td>{{ fmt(u.month.totalTokens) }}</td>
            <td>
              <span :class="u.disabled ? 'admin__badge admin__badge--off' : 'admin__badge admin__badge--on'">
                {{ u.disabled ? '已禁用' : '正常' }}
              </span>
            </td>
            <td class="admin__actions">
              <button class="btn btn--ghost" :disabled="busy" @click="showDetail(u)">用量</button>
              <button
                v-if="u.role !== 'admin'"
                class="btn btn--ghost"
                :class="{ 'btn--danger': !u.disabled }"
                :disabled="busy"
                @click="toggleDisable(u)"
              >
                {{ u.disabled ? '启用' : '禁用' }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>

      <!-- 用量明细 -->
      <div v-if="detail" class="admin__detail">
        <div class="admin__detail-head">
          <strong>{{ detail.user.username }}</strong>
          <span>{{ detail.user.planLabel }}</span>
          <span>建号 {{ detail.user.createdAt.slice(0, 10) }}</span>
          <button class="btn btn--ghost" @click="detail = null">收起</button>
        </div>
        <table class="admin__table admin__table--compact">
          <thead>
            <tr>
              <th>日期</th>
              <th>输入</th>
              <th>输出</th>
              <th>合计</th>
              <th>次数</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in detail.series.slice().reverse()" :key="d.day">
              <td>{{ String(d.day).slice(0, 4) }}-{{ String(d.day).slice(4, 6) }}-{{ String(d.day).slice(6, 8) }}</td>
              <td>{{ fmt(d.inputTokens) }}</td>
              <td>{{ fmt(d.outputTokens) }}</td>
              <td>{{ fmt(d.totalTokens) }}</td>
              <td>{{ d.calls }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- ============ 升级审核 ============ -->
    <div class="admin__block">
      <h3 class="admin__title">升级申请（待审核 {{ requests.length }}）</h3>
      <p v-if="requests.length === 0" class="admin__hint">没有待审核的申请。</p>
      <ul v-else class="admin__req-list">
        <li v-for="r in requests" :key="r.id" class="admin__req">
          <div class="admin__req-main">
            <strong>{{ r.username }}</strong>
            <span class="admin__req-plan">{{ r.toPlanLabel }}</span>
            <span class="admin__req-status">{{ STATUS_LABEL[r.status] }}</span>
            <span class="admin__req-time">{{ r.createdAt.slice(0, 16).replace('T', ' ') }}</span>
            <span v-if="r.note" class="admin__req-note">备注：{{ r.note }}</span>
          </div>
          <div class="admin__actions">
            <button class="btn" :disabled="busy" @click="review(r, true)">通过</button>
            <button class="btn btn--ghost btn--danger" :disabled="busy" @click="review(r, false)">
              驳回
            </button>
          </div>
        </li>
      </ul>
    </div>

    <!-- ============ 邀请码池 ============ -->
    <div class="admin__block">
      <h3 class="admin__title">邀请码池</h3>
      <div class="admin__gen">
        <label class="field">
          <span class="field__label">套餐</span>
          <select v-model="genPlanId" class="field__input">
            <option v-for="p in plans" :key="p.id" :value="p.id">{{ p.label }}</option>
          </select>
        </label>
        <label class="field">
          <span class="field__label">数量</span>
          <input v-model.number="genCount" class="field__input" type="number" min="1" max="200" />
        </label>
        <label class="field">
          <span class="field__label">批次名（可选）</span>
          <input v-model="genBatch" class="field__input" placeholder="比如 2026-双十一" />
        </label>
        <button class="btn" :disabled="busy || !genPlanId" @click="generate">生成</button>
      </div>

      <div v-if="generated.length" class="admin__generated">
        <div class="admin__generated-head">
          <span>刚生成的 {{ generated.length }} 个码：</span>
          <button class="btn btn--ghost" @click="copyGenerated">复制</button>
        </div>
        <pre class="admin__codes">{{ generated.join('\n') }}</pre>
      </div>

      <table class="admin__table admin__table--compact">
        <thead>
          <tr>
            <th>邀请码</th>
            <th>套餐</th>
            <th>批次</th>
            <th>状态</th>
            <th>使用者</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="c in codes" :key="c.code">
            <td><code>{{ c.code }}</code></td>
            <td>{{ plans.find((p) => p.id === c.planId)?.label ?? c.planId }}</td>
            <td>{{ c.batch ?? '—' }}</td>
            <td>
              <span
                :class="
                  c.usedBy
                    ? 'admin__badge admin__badge--used'
                    : c.revokedAt
                      ? 'admin__badge admin__badge--off'
                      : 'admin__badge admin__badge--on'
                "
              >
                {{ c.usedBy ? '已使用' : c.revokedAt ? '已作废' : '未使用' }}
              </span>
            </td>
            <td>{{ c.usedByName ?? '—' }}</td>
            <td class="admin__actions">
              <button
                v-if="!c.usedBy && !c.revokedAt"
                class="btn btn--ghost btn--danger"
                :disabled="busy"
                @click="revoke(c.code)"
              >
                作废
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
