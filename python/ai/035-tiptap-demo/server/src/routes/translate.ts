import { Router } from 'express'
import type { TranslateService } from '../services/translate.js'
import { isToneId } from '../config/defaults.js'
import { fail } from './httpError.js'

/**
 * 创建 /api/translate 路由。
 * 请求：{ text, source?: 'zh'|'en', target?: 'zh'|'en', tone?: ToneId }
 * 响应：{ translation }
 *
 * 默认 zh → en；en → zh 也支持（走另一个能力 / 另一套提示词）。
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
    // source / target：只认 zh 和 en，且方向必须是 zh→en 或 en→zh
    const source = body.source === undefined ? 'zh' : body.source
    const target = body.target === undefined ? 'en' : body.target
    if (source !== 'zh' && source !== 'en') {
      return res.status(400).json({ error: 'source must be "zh" or "en"' })
    }
    if (target !== 'zh' && target !== 'en') {
      return res.status(400).json({ error: 'target must be "zh" or "en"' })
    }
    if (source === target) {
      return res
        .status(400)
        .json({ error: 'source and target must differ (zh→en or en→zh)' })
    }
    if (body.tone !== undefined && !isToneId(body.tone)) {
      return res.status(400).json({ error: 'tone is not a supported tone id' })
    }

    // 2. 调 TranslateService
    try {
      const result = await service.translate(body.text.trim(), {
        source,
        target,
        tone: isToneId(body.tone) ? body.tone : undefined,
      })
      return res.json(result)
    } catch (err) {
      return fail(res, err)
    }
  })

  return router
}