import { Router } from 'express'
import type { TranslateService } from '../services/translate.js'
import { isToneId } from '../config/defaults.js'
import { fail } from './httpError.js'

/**
 * 创建 /api/translate 路由。
 * 请求：{ text, source?: 'zh', target?: 'en', tone?: ToneId }
 * 响应：{ translation }
 */
export function createTranslateRouter(service: TranslateService): Router {
  const router = Router()

  router.post('/api/translate', async (req, res) => {
    const body = (req.body ?? {}) as {
      text?: unknown
      source?: unknown
      target?: unknown
      tone?: unknown
    }

    // 1. 校验入参
    if (typeof body.text !== 'string' || !body.text.trim()) {
      return res.status(400).json({
        error: 'text is required and must be a non-empty string',
      })
    }
    // source/target 暂时只接受 zh→en，传其他值给 400 提示用户暂不支持
    if (body.source !== undefined && body.source !== 'zh') {
      return res.status(400).json({ error: 'only source="zh" is supported' })
    }
    if (body.target !== undefined && body.target !== 'en') {
      return res.status(400).json({ error: 'only target="en" is supported' })
    }
    if (body.tone !== undefined && !isToneId(body.tone)) {
      return res.status(400).json({ error: 'tone is not a supported tone id' })
    }

    // 2. 调 TranslateService
    try {
      const result = await service.translate(body.text.trim(), {
        tone: isToneId(body.tone) ? body.tone : undefined,
      })
      return res.json(result)
    } catch (err) {
      return fail(res, err)
    }
  })

  return router
}