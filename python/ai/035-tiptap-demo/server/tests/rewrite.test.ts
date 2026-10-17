import express from 'express'
import request from 'supertest'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createApp } from '../src/app.js'
import type { OllamaService } from '../src/services/ollama.js'

/**
 * 通过注入 mock service 打到真实路由层，验证：
 * - 入参校验
 * - service 调用参数透传
 * - 错误分类（500 解析失败 vs 502 上游失败）
 */

function buildMockService(impl?: OllamaService['rewriteWord']): OllamaService {
  return {
    rewriteWord:
      impl ??
      vi.fn(async () => ['great', 'promising', 'excellent']),
  }
}

describe('GET /healthz', () => {
  it('返回 200 + ok', async () => {
    const app = createApp({ service: buildMockService() })
    const res = await request(app).get('/healthz')
    expect(res.status).toBe(200)
    expect(res.body).toEqual({ ok: true })
  })
})

describe('POST /api/rewrite-word', () => {
  let app: ReturnType<typeof createApp>
  let service: OllamaService

  beforeEach(() => {
    service = buildMockService()
    app = createApp({ service })
  })

  it('正常调用：透传 word/context，返回 candidates', async () => {
    const spy = vi.spyOn(service, 'rewriteWord')
    const res = await request(app)
      .post('/api/rewrite-word')
      .send({ word: 'good', context: 'This is a good idea.' })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({
      candidates: ['great', 'promising', 'excellent'],
    })
    expect(spy).toHaveBeenCalledWith('good', { context: 'This is a good idea.' })
  })

  it('word 是空白字符串 → 400', async () => {
    const res = await request(app)
      .post('/api/rewrite-word')
      .send({ word: '   ' })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/word is required/)
  })

  it('缺 word 字段 → 400', async () => {
    const res = await request(app).post('/api/rewrite-word').send({})
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/word is required/)
  })

  it('word 不是字符串 → 400', async () => {
    const res = await request(app)
      .post('/api/rewrite-word')
      .send({ word: 123 })
    expect(res.status).toBe(400)
  })

  it('context 不是字符串 → 400', async () => {
    const res = await request(app)
      .post('/api/rewrite-word')
      .send({ word: 'good', context: 42 })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/context must be a string/)
  })

  it('service 抛 JSON 解析错误 → 500', async () => {
    app = createApp({
      service: buildMockService(async () => {
        throw new Error('cannot parse JSON array from model response')
      }),
    })
    const res = await request(app)
      .post('/api/rewrite-word')
      .send({ word: 'good' })
    expect(res.status).toBe(500)
    expect(res.body.error).toMatch(/JSON/)
  })

  it('service 抛上游错误 → 502', async () => {
    app = createApp({
      service: buildMockService(async () => {
        throw new Error('ECONNREFUSED 127.0.0.1:11434')
      }),
    })
    const res = await request(app)
      .post('/api/rewrite-word')
      .send({ word: 'good' })
    expect(res.status).toBe(502)
    expect(res.body.error).toMatch(/ECONNREFUSED/)
  })

  it('context 可省略', async () => {
    const spy = vi.spyOn(service, 'rewriteWord')
    const res = await request(app).post('/api/rewrite-word').send({ word: 'good' })
    expect(res.status).toBe(200)
    expect(spy).toHaveBeenCalledWith('good', { context: undefined })
  })

  it('word 自动 trim 后传给 service', async () => {
    const spy = vi.spyOn(service, 'rewriteWord')
    await request(app).post('/api/rewrite-word').send({ word: '  good  ' })
    expect(spy).toHaveBeenCalledWith('good', { context: undefined })
  })
})

// 确保 express 的类型导出可用（仅冒烟，避免 ambient module 漂移）
describe('express sanity', () => {
  it('express() 工厂可用', () => {
    expect(typeof express).toBe('function')
  })
})