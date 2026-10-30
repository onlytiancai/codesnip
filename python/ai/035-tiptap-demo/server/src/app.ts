import cors from 'cors'
import express, { type Express } from 'express'
import { createDictionaryRouter } from './routes/dictionary.js'
import { createRewriteRouter } from './routes/rewrite.js'
import { createSettingsRouter } from './routes/settings.js'
import { createTranslateRouter } from './routes/translate.js'
import type { ConfigStore } from './config/store.js'
import type { ClientRegistry } from './llm/registry.js'
import type { AiService } from './services/ai.js'
import type { TranslateService } from './services/translate.js'
import type { DictionaryService } from './services/dictionary.js'

export interface CreateAppOptions {
  aiService: AiService
  translateService: TranslateService
  /** 传了才挂载设置相关路由 */
  configStore?: ConfigStore
  registry?: ClientRegistry
  /** 传了才挂载查词相关路由 */
  dictionaryService?: DictionaryService
}

/**
 * 工厂：可注入 mock service，方便 supertest 直接打到路由层。
 */
export function createApp(options: CreateAppOptions): Express {
  const app = express()
  app.use(cors())
  app.use(express.json())
  app.use(createRewriteRouter(options.aiService))
  app.use(createTranslateRouter(options.translateService))
  if (options.configStore && options.registry) {
    app.use(
      createSettingsRouter({
        store: options.configStore,
        registry: options.registry,
      }),
    )
  }
  if (options.dictionaryService) {
    app.use(createDictionaryRouter({ dictionaryService: options.dictionaryService }))
  }

  app.get('/healthz', (_req, res) => {
    res.json({ ok: true })
  })

  return app
}
