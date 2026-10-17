import { Router } from 'express'
import type { OllamaService } from '../services/ollama.js'

/**
 * 创建 /api/rewrite-word 路由。生产里挂到 express app；测试里挂到 supertest。
 */
export function createRewriteRouter(service: OllamaService): Router {
  const router = Router()

  router.post('/api/rewrite-word', async (req, res) => {
    const body = (req.body ?? {}) as {
      word?: unknown
      context?: unknown
      sourceSentence?: unknown
    }

    // 1. 校验入参
    if (typeof body.word !== 'string' || !body.word.trim()) {
      return res.status(400).json({
        error: 'word is required and must be a non-empty string',
      })
    }
    if (body.context !== undefined && typeof body.context !== 'string') {
      return res.status(400).json({
        error: 'context must be a string when provided',
      })
    }
    if (body.sourceSentence !== undefined && typeof body.sourceSentence !== 'string') {
      return res.status(400).json({
        error: 'sourceSentence must be a string when provided',
      })
    }

    // 2. 调 Ollama service
    try {
      const candidates = await service.rewriteWord(body.word.trim(), {
        context: typeof body.context === 'string' ? body.context : undefined,
        sourceSentence:
          typeof body.sourceSentence === 'string' ? body.sourceSentence : undefined,
      })
      return res.json({ candidates })
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'unknown error'
      // 模型响应解析失败 → 500；Ollama 上游失败 → 502
      const isParseError =
        msg.includes('JSON') ||
        msg.includes('parse') ||
        msg.includes('array') ||
        msg.includes('empty response') ||
        msg.includes('no valid candidates')
      const status = isParseError ? 500 : 502
      return res.status(status).json({ error: msg })
    }
  })

  return router
}