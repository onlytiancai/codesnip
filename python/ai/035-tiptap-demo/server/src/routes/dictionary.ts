/**
 * 查词相关路由。
 *
 * 三个端点：
 * - GET  /api/dictionary/status    当前词库路径 + 索引状态（设置页「词库」tab 用）
 * - POST /api/dictionary/reload    重新读 .env + 重建索引
 * - GET  /api/dictionary/lookup    查一个单词
 *
 * 词典完全在本地，不调模型 —— 所以不经过 httpError.ts 的错误分类：
 * 「查不到这个词」是**正常结果**（200 + found:false），不是 404。
 */

import { Router } from 'express'
import type { DictionaryService } from '../services/dictionary.js'

export interface CreateDictionaryRouterOptions {
  dictionaryService: DictionaryService
}

export function createDictionaryRouter(
  options: CreateDictionaryRouterOptions,
): Router {
  const router = Router()
  const service = options.dictionaryService

  router.get('/api/dictionary/status', (_req, res) => {
    res.json(service.status())
  })

  router.post('/api/dictionary/reload', async (_req, res) => {
    try {
      res.json(await service.reload())
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'unknown error',
      })
    }
  })

  router.get('/api/dictionary/lookup', async (req, res) => {
    const raw = req.query.word
    const word = typeof raw === 'string' ? raw.trim() : ''
    if (!word) {
      return res.status(400).json({ error: 'word is required' })
    }

    const status = service.status()
    const entry = await service.lookup(word)
    if (!entry) {
      return res.json({
        found: false,
        word,
        error: status.loaded
          ? undefined
          : `词库未加载：${status.error ?? '未知原因'}`,
      })
    }
    return res.json({ found: true, word, entry })
  })

  return router
}
