import request from 'supertest'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createApp } from '../src/app.js'
import type { OllamaService } from '../src/services/ollama.js'
import type { TranslateService } from '../src/services/translate.js'

/**
 * /api/translate 路由层集成测试：注入 mock service，验证入参校验、透传、错误分类。
 */

function buildRewriteService(): OllamaService {
  return {
    rewriteWord: vi.fn(async () => ['x']),
  }
}

function buildTranslateService(impl?: TranslateService['translate']): TranslateService {
  return {
    translate:
      impl ??
      vi.fn(async () => ({
        translation: 'I went to the park yesterday.',
      })),
  }
}

function buildApp(translateService: TranslateService): ReturnType<typeof createApp> {
  return createApp({
    rewriteService: buildRewriteService(),
    translateService,
  })
}

describe('POST /api/translate', () => {
  let app: ReturnType<typeof createApp>
  let service: TranslateService

  beforeEach(() => {
    service = buildTranslateService()
    app = buildApp(service)
  })

  it('正常调用：返回 translation', async () => {
    const spy = vi.spyOn(service, 'translate')
    const res = await request(app)
      .post('/api/translate')
      .send({ text: '我昨天去公园。' })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({
      translation: 'I went to the park yesterday.',
    })
    expect(spy).toHaveBeenCalledWith('我昨天去公园。')
  })

  it('允许显式 source="zh" target="en"', async () => {
    const spy = vi.spyOn(service, 'translate')
    const res = await request(app)
      .post('/api/translate')
      .send({ text: '你好', source: 'zh', target: 'en' })
    expect(res.status).toBe(200)
    expect(spy).toHaveBeenCalledWith('你好')
  })

  it('缺 text → 400', async () => {
    const res = await request(app).post('/api/translate').send({})
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/text is required/)
  })

  it('text 是空白字符串 → 400', async () => {
    const res = await request(app)
      .post('/api/translate')
      .send({ text: '   ' })
    expect(res.status).toBe(400)
  })

  it('text 不是字符串 → 400', async () => {
    const res = await request(app)
      .post('/api/translate')
      .send({ text: 42 })
    expect(res.status).toBe(400)
  })

  it('source 不是 zh → 400', async () => {
    const res = await request(app)
      .post('/api/translate')
      .send({ text: 'hi', source: 'en' })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/only source="zh"/)
  })

  it('target 不是 en → 400', async () => {
    const res = await request(app)
      .post('/api/translate')
      .send({ text: 'hi', target: 'fr' })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/only target="en"/)
  })

  it('service 抛"empty translation"错误 → 500', async () => {
    app = buildApp(
      buildTranslateService(async () => {
        throw new Error('empty translation after parsing')
      }),
    )
    const res = await request(app)
      .post('/api/translate')
      .send({ text: '你好' })
    expect(res.status).toBe(500)
    expect(res.body.error).toMatch(/empty translation/)
  })

  it('service 抛上游错误 → 502', async () => {
    app = buildApp(
      buildTranslateService(async () => {
        throw new Error('ECONNREFUSED 127.0.0.1:11434')
      }),
    )
    const res = await request(app)
      .post('/api/translate')
      .send({ text: '你好' })
    expect(res.status).toBe(502)
    expect(res.body.error).toMatch(/ECONNREFUSED/)
  })

  it('text 自动 trim 后传给 service', async () => {
    const spy = vi.spyOn(service, 'translate')
    await request(app).post('/api/translate').send({ text: '  你好  ' })
    expect(spy).toHaveBeenCalledWith('你好')
  })
})