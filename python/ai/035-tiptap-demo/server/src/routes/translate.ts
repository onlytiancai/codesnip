import { Router } from 'express'
import type { TranslateService } from '../services/translate.js'

/**
 * 创建 /api/translate 路由。
 * 请求：{ text, source?: 'zh', target?: 'en' }
 * 响应：{ translation, sentences: [{ zh, en }] }
 */
export function createTranslateRouter(service: TranslateService): Router {
  const router = Router()

  router.post('/api/translate', async (req, res) => {
    const body = (req.body ?? {}) as { text?: unknown; source?: unknown; target?: unknown }

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

    // 2. 调 TranslateService
    try {
      const result = await service.translate(body.text.trim())
      return res.json(result)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'unknown error'
      // 区分"模型响应解析失败" vs "上游 Ollama 调用失败"
      const isParseError =
        msg.includes('JSON') ||
        msg.includes('parse') ||
        msg.includes('object') ||
        msg.includes('array') ||
        msg.includes('missing') ||
        msg.includes('empty') ||
        msg.includes('no valid')
      const status = isParseError ? 500 : 502
      return res.status(status).json({ error: msg })
    }
  })

  return router
}