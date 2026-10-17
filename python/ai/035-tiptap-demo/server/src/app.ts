import cors from 'cors'
import express, { type Express } from 'express'
import { createRewriteRouter } from './routes/rewrite.js'
import { createTranslateRouter } from './routes/translate.js'
import type { OllamaService } from './services/ollama.js'
import type { TranslateService } from './services/translate.js'

export interface CreateAppOptions {
  rewriteService: OllamaService
  translateService: TranslateService
}

/**
 * 工厂：可注入 mock service，方便 supertest 直接打到路由层。
 */
export function createApp(options: CreateAppOptions): Express {
  const app = express()
  app.use(cors())
  app.use(express.json())
  app.use(createRewriteRouter(options.rewriteService))
  app.use(createTranslateRouter(options.translateService))

  app.get('/healthz', (_req, res) => {
    res.json({ ok: true })
  })

  return app
}