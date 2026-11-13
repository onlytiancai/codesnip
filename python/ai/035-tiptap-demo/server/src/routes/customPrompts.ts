/**
 * 用户自定义指令（用户提示词）的路由。
 *
 * 和「系统提示词」的区别：
 *  - 系统提示词（config/store.ts 那套）是**管理员**维护的全局配置，
 *    所有人共用，改不了。
 *  - 用户自定义指令是**每个用户自己**的，每个 provider 一套，
 *    跑起来时套在系统提示词（rewriteInstruction 能力）后面。
 *
 * 谁能用：
 *  - 只有付费用户能新建/修改（免费版看得到自己的指令但不能改）
 *  - 管理员不受限
 *  - 指令只有本人能看 / 改 / 删 / 跑
 */

import { Router } from 'express'
import type { AiService } from '../services/ai.js'
import type { ConfigStore } from '../config/store.js'
import type {
  UserPrompt,
  UserPromptScope,
  UserStore,
} from '../config/userStore.js'
import { currentUser } from '../services/requestContext.js'
import { requireLogin, requireUser } from './auth.js'
import { isPaidPlan } from '../config/plans.js'
import { fail } from './httpError.js'

const SCOPES: UserPromptScope[] = ['sentence', 'paragraph', 'full']

/** 作用范围的中文说明（前端也用一份，见 constants/customPrompts.ts）。 */
export const SCOPE_LABELS: Record<UserPromptScope, string> = {
  sentence: '句子',
  paragraph: '段落',
  full: '全文',
}

/** 对外的形状 —— 不带 userId（那是内部的东西）。 */
export interface UserPromptInfo {
  id: string
  providerId: string
  label: string
  icon: string
  scope: UserPromptScope
  instruction: string
  showInToolbar: boolean
  ordinal: number
  createdAt: string
  updatedAt: string
}

function publicPrompt(p: UserPrompt): UserPromptInfo {
  return {
    id: p.id,
    providerId: p.providerId,
    label: p.label,
    icon: p.icon,
    scope: p.scope,
    instruction: p.instruction,
    showInToolbar: p.showInToolbar,
    ordinal: p.ordinal,
    createdAt: p.createdAt,
    updatedAt: p.updatedAt,
  }
}

interface ParsedPrompt {
  label: string
  icon: string
  scope: UserPromptScope
  instruction: string
  showInToolbar: boolean
}

function parsePrompt(body: unknown): ParsedPrompt | { error: string } {
  const b = (body ?? {}) as Record<string, unknown>

  const label = typeof b.label === 'string' ? b.label.trim() : ''
  if (!label) return { error: 'name is required' }
  if (label.length > 40) return { error: '名字太长了（最多 40 字）' }

  const icon = typeof b.icon === 'string' && b.icon.trim() ? b.icon.trim() : '✨'
  if (icon.length > 8) return { error: '图标太长了（最多 8 个字符）' }

  const scope = b.scope
  if (!SCOPES.includes(scope as UserPromptScope)) {
    return { error: `scope must be one of ${SCOPES.join(' | ')}` }
  }

  const instruction = typeof b.instruction === 'string' ? b.instruction.trim() : ''
  if (!instruction) return { error: 'instruction is required' }
  if (instruction.length > 2000) return { error: '指令太长了（最多 2000 字）' }

  return {
    label,
    icon,
    scope: scope as UserPromptScope,
    instruction,
    showInToolbar: b.showInToolbar !== false,
  }
}

export interface CreateCustomPromptsRouterOptions {
  userStore: UserStore
  aiService: AiService
  /** 用来知道「当前生效的 provider」—— 用户指令是跟着 provider 走的 */
  store?: ConfigStore
}

export function createCustomPromptsRouter(
  options: CreateCustomPromptsRouterOptions,
): Router {
  const { userStore, aiService } = options
  const router = Router()
  const login = requireLogin(userStore)
  const user = requireUser(userStore)

  /** 当前登录用户；没登录 / 本地工具 → null。 */
  function actor(): { id: number; role: string; planId: string } | null {
    const u = currentUser()
    if (!u) return null
    return { id: u.id, role: u.role ?? 'user', planId: u.planId ?? 'free' }
  }

  // ---- 便捷端点：当前生效的 provider ----
  //
  // 编辑器里要显示「我的指令」，但它得跟着**当前生效的 provider** 走 ——
  // 这两个端点省得前端先查一遍生效的是谁。

  router.get('/api/custom-prompts', login, (_req, res) => {
    const me = actor()
    if (!me) return res.status(401).json({ error: '需要登录' })
    const providerId = options.store?.getActiveProviderId() ?? ''
    if (!providerId) {
      return res.json({ activeProviderId: '', prompts: [] })
    }
    try {
      res.json({
        activeProviderId: providerId,
        prompts: userStore.listUserPrompts(me.id, providerId).map(publicPrompt),
      })
    } catch (err) {
      return fail(res, err)
    }
  })

  router.post('/api/custom-prompts', login, (req, res) => {
    const me = actor()
    if (!me) return res.status(401).json({ error: '需要登录' })
    if (me.role !== 'admin' && !isPaidPlan(me.planId)) {
      return res.status(403).json({
        error: '自定义指令是付费功能 —— 升级套餐后就能用了',
      })
    }
    const providerId = options.store?.getActiveProviderId() ?? ''
    if (!providerId) {
      return res.status(400).json({ error: '当前没有生效的 provider' })
    }
    const parsed = parsePrompt(req.body)
    if ('error' in parsed) {
      return res.status(400).json({ error: parsed.error })
    }
    try {
      const created = userStore.createUserPrompt({
        userId: me.id,
        providerId,
        ...parsed,
      })
      res.json({ prompt: publicPrompt(created) })
    } catch (err) {
      return fail(res, err)
    }
  })

  // ---- 列表（自己的）----

  router.get('/api/providers/:id/custom-prompts', login, (req, res) => {
    const me = actor()
    if (!me) return res.status(401).json({ error: '需要登录' })
    try {
      const list = userStore
        .listUserPrompts(me.id, req.params.id)
        .map(publicPrompt)
      res.json({ prompts: list })
    } catch (err) {
      return fail(res, err)
    }
  })

  // ---- 新建 ----

  router.post('/api/providers/:id/custom-prompts', login, (req, res) => {
    const me = actor()
    if (!me) return res.status(401).json({ error: '需要登录' })
    // 免费版只能看，不能建
    if (me.role !== 'admin' && !isPaidPlan(me.planId)) {
      return res.status(403).json({
        error: '自定义指令是付费功能 —— 升级套餐后就能用了',
      })
    }
    const parsed = parsePrompt(req.body)
    if ('error' in parsed) {
      return res.status(400).json({ error: parsed.error })
    }
    try {
      const created = userStore.createUserPrompt({
        userId: me.id,
        providerId: req.params.id,
        ...parsed,
      })
      res.json({ prompt: publicPrompt(created) })
    } catch (err) {
      return fail(res, err)
    }
  })

  // ---- 改 ----

  router.put('/api/custom-prompts/:id', login, (req, res) => {
    const me = actor()
    if (!me) return res.status(401).json({ error: '需要登录' })
    const existing = userStore.getUserPrompt(req.params.id)
    if (!existing || existing.userId !== me.id) {
      return res.status(404).json({ error: '指令不存在' })
    }
    const parsed = parsePrompt(req.body)
    if ('error' in parsed) {
      return res.status(400).json({ error: parsed.error })
    }
    const body = (req.body ?? {}) as Record<string, unknown>
    const ordinal =
      typeof body.ordinal === 'number' && Number.isFinite(body.ordinal)
        ? Math.floor(body.ordinal)
        : undefined
    try {
      const updated = userStore.updateUserPrompt(req.params.id, {
        ...parsed,
        ...(ordinal === undefined ? {} : { ordinal }),
      })
      res.json({ prompt: publicPrompt(updated!) })
    } catch (err) {
      return fail(res, err)
    }
  })

  // ---- 删 ----

  router.delete('/api/custom-prompts/:id', login, (req, res) => {
    const me = actor()
    if (!me) return res.status(401).json({ error: '需要登录' })
    const existing = userStore.getUserPrompt(req.params.id)
    if (!existing || existing.userId !== me.id) {
      return res.status(404).json({ error: '指令不存在' })
    }
    try {
      userStore.deleteUserPrompt(req.params.id)
      res.json({ ok: true })
    } catch (err) {
      return fail(res, err)
    }
  })

  // ---- 跑一条指令 ----
  //
  // 指令文本从库里取（不信客户端传什么），
  // 客户端只负责把「按作用范围取好的文本」送过来。

  router.post('/api/custom-prompts/:id/run', user, async (req, res) => {
    const me = actor()
    if (!me) return res.status(401).json({ error: '需要登录' })
    const prompt = userStore.getUserPrompt(req.params.id)
    if (!prompt || prompt.userId !== me.id) {
      return res.status(404).json({ error: '指令不存在' })
    }
    const body = (req.body ?? {}) as {
      text?: unknown
      context?: unknown
      sourceSentence?: unknown
      /** 「继续调整」时补的一句话 —— 拼在用户指令后面 */
      extraInstruction?: unknown
    }
    if (typeof body.text !== 'string' || !body.text.trim()) {
      return res
        .status(400)
        .json({ error: 'text is required and must be a non-empty string' })
    }
    if (
      body.context !== undefined &&
      typeof body.context !== 'string'
    ) {
      return res.status(400).json({ error: 'context must be a string' })
    }
    if (
      body.sourceSentence !== undefined &&
      typeof body.sourceSentence !== 'string'
    ) {
      return res.status(400).json({ error: 'sourceSentence must be a string' })
    }
    if (
      body.extraInstruction !== undefined &&
      typeof body.extraInstruction !== 'string'
    ) {
      return res.status(400).json({ error: 'extraInstruction must be a string' })
    }
    const extra =
      typeof body.extraInstruction === 'string'
        ? body.extraInstruction.trim()
        : ''
    const instruction = extra
      ? `${prompt.instruction}\n${extra}`
      : prompt.instruction
    try {
      const text = await aiService.rewriteInstruction(body.text.trim(), {
        instruction,
        context: typeof body.context === 'string' ? body.context : undefined,
        sourceSentence:
          typeof body.sourceSentence === 'string'
            ? body.sourceSentence
            : undefined,
      })
      res.json({ text })
    } catch (err) {
      return fail(res, err)
    }
  })

  return router
}
