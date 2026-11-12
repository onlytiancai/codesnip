import { Router } from 'express'
import type { AiService } from '../services/ai.js'
import { fail } from './httpError.js'
import { isSelectionKind } from '../prompts/rewriteSelection.js'
import {
  isSentenceOperation,
  type SentenceOperation,
} from '../prompts/rewriteSentence.js'
import { isToneId, type ToneId } from '../config/defaults.js'

/**
 * AI 改写相关的四个路由：
 *   POST /api/rewrite-selection     单词 / 短语 → 候选数组
 *   POST /api/rewrite-sentence      整句 rewrite/improve/shorten/expand
 *   POST /api/grammar-check         语法 / 标点检查（只提示）
 *   POST /api/rewrite-instruction   自定义指令改写
 *
 * 生产里挂到 express app；测试里挂到 supertest。
 */

/** 可选字符串字段的公共校验：undefined 放行，非字符串 400。 */
function checkOptionalString(
  value: unknown,
  name: string,
): { error?: string } {
  if (value !== undefined && typeof value !== 'string') {
    return { error: `${name} must be a string when provided` }
  }
  return {}
}

export function createRewriteRouter(service: AiService): Router {
  const router = Router()

  // ---- 选区级：单词 / 短语 → 候选 ----
  router.post('/api/rewrite-selection', async (req, res) => {
    const body = (req.body ?? {}) as {
      text?: unknown
      kind?: unknown
      context?: unknown
      sourceSentence?: unknown
    }

    if (typeof body.text !== 'string' || !body.text.trim()) {
      return res
        .status(400)
        .json({ error: 'text is required and must be a non-empty string' })
    }
    if (!isSelectionKind(body.kind)) {
      return res
        .status(400)
        .json({ error: 'kind is required and must be "word" or "phrase"' })
    }
    const ctxErr = checkOptionalString(body.context, 'context')
    if (ctxErr.error) return res.status(400).json({ error: ctxErr.error })
    const srcErr = checkOptionalString(body.sourceSentence, 'sourceSentence')
    if (srcErr.error) return res.status(400).json({ error: srcErr.error })

    try {
      const candidates = await service.rewriteSelection(body.text.trim(), body.kind, {
        context: typeof body.context === 'string' ? body.context : undefined,
        sourceSentence:
          typeof body.sourceSentence === 'string' ? body.sourceSentence : undefined,
      })
      return res.json({ candidates })
    } catch (err) {
      return fail(res, err)
    }
  })

  // ---- 整句改写 ----
  router.post('/api/rewrite-sentence', async (req, res) => {
    const body = (req.body ?? {}) as {
      sentence?: unknown
      operation?: unknown
      tone?: unknown
      context?: unknown
      sourceSentence?: unknown
    }

    if (typeof body.sentence !== 'string' || !body.sentence.trim()) {
      return res
        .status(400)
        .json({ error: 'sentence is required and must be a non-empty string' })
    }
    if (!isSentenceOperation(body.operation)) {
      return res.status(400).json({
        error:
          'operation is required and must be one of rewrite, improve, shorten, expand',
      })
    }
    if (body.tone !== undefined && !isToneId(body.tone)) {
      return res.status(400).json({ error: 'tone is not a supported tone id' })
    }
    const ctxErr = checkOptionalString(body.context, 'context')
    if (ctxErr.error) return res.status(400).json({ error: ctxErr.error })
    const srcErr = checkOptionalString(body.sourceSentence, 'sourceSentence')
    if (srcErr.error) return res.status(400).json({ error: srcErr.error })

    try {
      const text = await service.rewriteSentence(body.sentence.trim(), {
        operation: body.operation,
        tone: isToneId(body.tone) ? (body.tone as ToneId) : undefined,
        context: typeof body.context === 'string' ? body.context : undefined,
        sourceSentence:
          typeof body.sourceSentence === 'string' ? body.sourceSentence : undefined,
      })
      return res.json({ text })
    } catch (err) {
      return fail(res, err)
    }
  })

  // ---- 语法 / 标点检查 ----
  router.post('/api/grammar-check', async (req, res) => {
    const body = (req.body ?? {}) as { sentence?: unknown }

    if (typeof body.sentence !== 'string' || !body.sentence.trim()) {
      return res
        .status(400)
        .json({ error: 'sentence is required and must be a non-empty string' })
    }

    try {
      // 模型原文直接透传，前端原样显示，不做结构化解析
      const text = await service.grammarCheck(body.sentence.trim())
      return res.json({ text })
    } catch (err) {
      return fail(res, err)
    }
  })

  // ---- 自定义指令改写 ----
  router.post('/api/rewrite-instruction', async (req, res) => {
    const body = (req.body ?? {}) as {
      text?: unknown
      instruction?: unknown
      context?: unknown
    }

    if (typeof body.text !== 'string' || !body.text.trim()) {
      return res
        .status(400)
        .json({ error: 'text is required and must be a non-empty string' })
    }
    if (typeof body.instruction !== 'string' || !body.instruction.trim()) {
      return res
        .status(400)
        .json({ error: 'instruction is required and must be a non-empty string' })
    }
    const ctxErr = checkOptionalString(body.context, 'context')
    if (ctxErr.error) return res.status(400).json({ error: ctxErr.error })

    try {
      const text = await service.rewriteInstruction(body.text.trim(), {
        instruction: body.instruction.trim(),
        context: typeof body.context === 'string' ? body.context : undefined,
      })
      return res.json({ text })
    } catch (err) {
      return fail(res, err)
    }
  })

  return router
}

export type { SentenceOperation }
