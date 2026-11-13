import cors from 'cors'
import express, { type Express } from 'express'
import {
  createAuthRouter,
  PROTECTED_API_PATHS,
  requireUser,
} from './routes/auth.js'
import { createAdminRouter } from './routes/admin.js'
import { createDictionaryRouter } from './routes/dictionary.js'
import { createPlansRouter } from './routes/plans.js'
import { createRewriteRouter } from './routes/rewrite.js'
import { createSettingsRouter } from './routes/settings.js'
import { createCustomPromptsRouter } from './routes/customPrompts.js'
import { createTranslateRouter } from './routes/translate.js'
import type { ConfigStore } from './config/store.js'
import type { UserStore } from './config/userStore.js'
import type { ClientRegistry } from './llm/registry.js'
import type { AiService } from './services/ai.js'
import type { TranslateService } from './services/translate.js'
import type { DictionaryService } from './services/dictionary.js'
import { debugEnabled } from './utils/debug.js'

export interface CreateAppOptions {
  aiService: AiService
  translateService: TranslateService
  /** 传了才挂载设置相关路由 */
  configStore?: ConfigStore
  registry?: ClientRegistry
  /** 传了才挂载查词相关路由 */
  dictionaryService?: DictionaryService
  /** 传了才挂载登录相关路由 + 「消耗 token 的接口」的登录守卫 */
  userStore?: UserStore
  /** 记录 token 用量（不传就不记） */
  recordUsage?: (usage: { inputTokens: number; outputTokens: number }) => void
}

/**
 * 工厂：可注入 mock service，方便 supertest 直接打到路由层。
 */
export function createApp(options: CreateAppOptions): Express {
  const app = express()
  app.use(cors())
  app.use(express.json())

  // 登录相关：注册 / 登录 / 登出 / 当前用户
  // —— 传了 userStore 才启用鉴权（不传 = 全部放行，单元测试靠这个绕开登录）
  if (options.userStore) {
    app.use(createAuthRouter({ userStore: options.userStore }))
    const guard = requireUser(options.userStore)
    // 「消耗 token 的接口」挂守卫 —— 必须在下面的 router 之前
    for (const path of PROTECTED_API_PATHS) {
      app.use(path, guard)
    }
    // 套餐（申请 / 兑换）和管理端 —— 守卫挂在各自 router 内部，
    // 免得把「只要登录就行」和「查配额」的规则搞混
    app.use(createPlansRouter({ userStore: options.userStore }))
    app.use(createAdminRouter({ userStore: options.userStore }))
  }

  app.use(createRewriteRouter(options.aiService))
  app.use(createTranslateRouter(options.translateService))
  if (options.configStore && options.registry) {
    app.use(
      createSettingsRouter({
        store: options.configStore,
        registry: options.registry,
        recordUsage: options.recordUsage,
        // 传了 userStore → 「改配置」这类接口只有管理员能动
        userStore: options.userStore,
      }),
    )
  }
  if (options.dictionaryService) {
    app.use(
      createDictionaryRouter({
        dictionaryService: options.dictionaryService,
        userStore: options.userStore,
      }),
    )
  }
  // 用户自定义指令（用户提示词）—— 需要登录 + 身份
  if (options.userStore) {
    app.use(
      createCustomPromptsRouter({
        userStore: options.userStore,
        aiService: options.aiService,
        store: options.configStore,
      }),
    )
  }

  // 前端拿这个开关决定要不要打 console.log（和 server/.env 的 DEBUG 同一个）
  app.get('/api/debug', (_req, res) => {
    res.json({ enabled: debugEnabled() })
  })

  app.get('/healthz', (_req, res) => {
    res.json({ ok: true })
  })

  return app
}
