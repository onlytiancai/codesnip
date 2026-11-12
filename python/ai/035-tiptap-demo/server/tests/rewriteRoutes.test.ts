import request from 'supertest'
import { describe, expect, it, vi } from 'vitest'
import { buildAiService, buildApp } from './helpers.js'
import type { AiService } from '../src/services/ai.js'

/**
 * 四个 AI 端点的路由层测试：入参校验、参数透传、错误分类。
 */

describe('GET /healthz', () => {
  it('返回 200 + ok', async () => {
    const res = await request(buildApp()).get('/healthz')
    expect(res.status).toBe(200)
    expect(res.body).toEqual({ ok: true })
  })
})

describe('POST /api/rewrite-selection', () => {
  it('透传 text/kind/context/sourceSentence，返回 candidates', async () => {
    const svc = buildAiService()
    const res = await request(buildApp(svc))
      .post('/api/rewrite-selection')
      .send({
        text: 'very important',
        kind: 'phrase',
        context: 'Education is very important for children.',
        sourceSentence: '教育对孩子很重要。',
      })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({ candidates: ['great', 'promising', 'excellent'] })
    expect(svc.rewriteSelection).toHaveBeenCalledWith('very important', 'phrase', {
      context: 'Education is very important for children.',
      sourceSentence: '教育对孩子很重要。',
    })
  })

  it('context / sourceSentence 可省略', async () => {
    const svc = buildAiService()
    const res = await request(buildApp(svc))
      .post('/api/rewrite-selection')
      .send({ text: 'good', kind: 'word' })

    expect(res.status).toBe(200)
    expect(svc.rewriteSelection).toHaveBeenCalledWith('good', 'word', {
      context: undefined,
      sourceSentence: undefined,
    })
  })

  it('text 自动 trim', async () => {
    const svc = buildAiService()
    await request(buildApp(svc))
      .post('/api/rewrite-selection')
      .send({ text: '  good  ', kind: 'word' })
    expect(svc.rewriteSelection).toHaveBeenCalledWith('good', 'word', expect.anything())
  })

  it('缺 text → 400', async () => {
    const res = await request(buildApp())
      .post('/api/rewrite-selection')
      .send({ kind: 'word' })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/text is required/)
  })

  it('text 是空白 → 400', async () => {
    const res = await request(buildApp())
      .post('/api/rewrite-selection')
      .send({ text: '  ', kind: 'word' })
    expect(res.status).toBe(400)
  })

  it('缺 kind → 400', async () => {
    const res = await request(buildApp())
      .post('/api/rewrite-selection')
      .send({ text: 'good' })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/kind is required/)
  })

  it('kind 非法 → 400', async () => {
    const res = await request(buildApp())
      .post('/api/rewrite-selection')
      .send({ text: 'good', kind: 'sentence' })
    expect(res.status).toBe(400)
  })

  it('context / sourceSentence 非字符串 → 400', async () => {
    const a = await request(buildApp())
      .post('/api/rewrite-selection')
      .send({ text: 'g', kind: 'word', context: 42 })
    expect(a.status).toBe(400)
    expect(a.body.error).toMatch(/context must be a string/)

    const b = await request(buildApp())
      .post('/api/rewrite-selection')
      .send({ text: 'g', kind: 'word', sourceSentence: { obj: true } })
    expect(b.status).toBe(400)
    expect(b.body.error).toMatch(/sourceSentence must be a string/)
  })

  it('模型解析失败 → 500', async () => {
    const svc = buildAiService({
      rewriteSelection: vi.fn(async () => {
        throw new Error('cannot parse JSON array from model response')
      }),
    })
    const res = await request(buildApp(svc))
      .post('/api/rewrite-selection')
      .send({ text: 'good', kind: 'word' })
    expect(res.status).toBe(500)
    expect(res.body.error).toMatch(/JSON/)
  })

  it('上游失败 → 502', async () => {
    const svc = buildAiService({
      rewriteSelection: vi.fn(async () => {
        throw new Error('ECONNREFUSED 127.0.0.1:11434')
      }),
    })
    const res = await request(buildApp(svc))
      .post('/api/rewrite-selection')
      .send({ text: 'good', kind: 'word' })
    expect(res.status).toBe(502)
    expect(res.body.error).toMatch(/ECONNREFUSED/)
  })
})

describe('POST /api/rewrite-sentence', () => {
  it('透传 sentence/operation/tone，返回 text', async () => {
    const svc = buildAiService()
    const res = await request(buildApp(svc))
      .post('/api/rewrite-sentence')
      .send({
        sentence: 'Education is very important for children.',
        operation: 'rewrite',
        tone: 'formal',
        sourceSentence: '教育对孩子很重要。',
      })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({ text: 'Education is of great importance to children.' })
    expect(svc.rewriteSentence).toHaveBeenCalledWith(
      'Education is very important for children.',
      {
        operation: 'rewrite',
        tone: 'formal',
        context: undefined,
        sourceSentence: '教育对孩子很重要。',
      },
    )
  })

  it('tone 可省略 → undefined', async () => {
    const svc = buildAiService()
    await request(buildApp(svc))
      .post('/api/rewrite-sentence')
      .send({ sentence: 'x.', operation: 'improve' })
    expect(svc.rewriteSentence).toHaveBeenCalledWith('x.', expect.objectContaining({ tone: undefined }))
  })

  it('四个 operation 都接受', async () => {
    for (const op of ['rewrite', 'improve', 'shorten', 'expand']) {
      const res = await request(buildApp())
        .post('/api/rewrite-sentence')
        .send({ sentence: 'x.', operation: op })
      expect(res.status).toBe(200)
    }
  })

  it('缺 sentence → 400', async () => {
    const res = await request(buildApp())
      .post('/api/rewrite-sentence')
      .send({ operation: 'rewrite' })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/sentence is required/)
  })

  it('operation 缺失或非法 → 400', async () => {
    const a = await request(buildApp())
      .post('/api/rewrite-sentence')
      .send({ sentence: 'x.' })
    expect(a.status).toBe(400)

    const b = await request(buildApp())
      .post('/api/rewrite-sentence')
      .send({ sentence: 'x.', operation: 'polish' })
    expect(b.status).toBe(400)
  })

  it('tone 非法 → 400', async () => {
    const res = await request(buildApp())
      .post('/api/rewrite-sentence')
      .send({ sentence: 'x.', operation: 'rewrite', tone: 'sarcastic' })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/tone/)
  })

  it('上游失败 → 502', async () => {
    const svc = buildAiService({
      rewriteSentence: vi.fn(async () => {
        throw new Error('ECONNREFUSED')
      }),
    })
    const res = await request(buildApp(svc))
      .post('/api/rewrite-sentence')
      .send({ sentence: 'x.', operation: 'rewrite' })
    expect(res.status).toBe(502)
  })
})

describe('POST /api/grammar-check', () => {
  it('原样返回模型文本', async () => {
    const raw = '「go」应该改成「goes」，因为主语 He 是第三人称单数。'
    const svc = buildAiService({ grammarCheck: vi.fn(async () => raw) })
    const res = await request(buildApp(svc))
      .post('/api/grammar-check')
      .send({ sentence: 'He go to school.' })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({ text: raw })
    expect(svc.grammarCheck).toHaveBeenCalledWith('He go to school.')
  })

  it('模型说没错就把那句提示原样返回', async () => {
    const res = await request(buildApp())
      .post('/api/grammar-check')
      .send({ sentence: 'He goes to school.' })
    expect(res.status).toBe(200)
    expect(res.body).toEqual({ text: '这句话没有语法或标点错误。' })
  })

  it('缺 sentence → 400', async () => {
    const res = await request(buildApp()).post('/api/grammar-check').send({})
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/sentence is required/)
  })

  it('上游失败 → 500', async () => {
    const svc = buildAiService({
      grammarCheck: vi.fn(async () => {
        throw new Error('empty response')
      }),
    })
    const res = await request(buildApp(svc))
      .post('/api/grammar-check')
      .send({ sentence: 'x.' })
    expect(res.status).toBe(500)
  })

  it('上游失败 → 502', async () => {
    const svc = buildAiService({
      grammarCheck: vi.fn(async () => {
        throw new Error('ECONNREFUSED')
      }),
    })
    const res = await request(buildApp(svc))
      .post('/api/grammar-check')
      .send({ sentence: 'x.' })
    expect(res.status).toBe(502)
  })
})

describe('POST /api/rewrite-instruction', () => {
  it('透传 text/instruction/context，返回 text', async () => {
    const svc = buildAiService()
    const res = await request(buildApp(svc))
      .post('/api/rewrite-instruction')
      .send({
        text: 'Education is very important for children.',
        instruction: 'Make it more academic.',
        context: 'First sentence.',
      })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({ text: 'Rewritten text.' })
    expect(svc.rewriteInstruction).toHaveBeenCalledWith(
      'Education is very important for children.',
      { instruction: 'Make it more academic.', context: 'First sentence.' },
    )
  })

  it('instruction 自动 trim', async () => {
    const svc = buildAiService()
    await request(buildApp(svc))
      .post('/api/rewrite-instruction')
      .send({ text: 'x.', instruction: '  Simplify it.  ' })
    expect(svc.rewriteInstruction).toHaveBeenCalledWith(
      'x.',
      expect.objectContaining({ instruction: 'Simplify it.' }),
    )
  })

  it('缺 text → 400', async () => {
    const res = await request(buildApp())
      .post('/api/rewrite-instruction')
      .send({ instruction: 'do it' })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/text is required/)
  })

  it('缺 instruction → 400', async () => {
    const res = await request(buildApp())
      .post('/api/rewrite-instruction')
      .send({ text: 'x.' })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/instruction is required/)
  })

  it('context 非字符串 → 400', async () => {
    const res = await request(buildApp())
      .post('/api/rewrite-instruction')
      .send({ text: 'x.', instruction: 'y', context: 5 })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/context must be a string/)
  })

  it('上游失败 → 502', async () => {
    const svc: AiService = buildAiService({
      rewriteInstruction: vi.fn(async () => {
        throw new Error('ECONNREFUSED')
      }),
    })
    const res = await request(buildApp(svc))
      .post('/api/rewrite-instruction')
      .send({ text: 'x.', instruction: 'y' })
    expect(res.status).toBe(502)
  })
})
