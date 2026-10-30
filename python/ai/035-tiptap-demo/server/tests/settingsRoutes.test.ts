import request from 'supertest'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { createApp } from '../src/app.js'
import {
  buildAiService,
  buildRegistry,
  buildTempStore,
  buildTranslateService,
  type TempStore,
} from './helpers.js'
import type { ConfigStore } from '../src/config/store.js'

/**
 * 设置路由：provider 管理 + 每个 provider 各自的提示词（版本 / 回滚 / 试运行）。
 */

let tmp: TempStore
let store: ConfigStore
let app: ReturnType<typeof createApp>

beforeEach(() => {
  tmp = buildTempStore()
  store = tmp.store
  app = createApp({
    aiService: buildAiService(),
    translateService: buildTranslateService(),
    configStore: store,
    registry: buildRegistry('OK'),
  })
})

afterEach(() => {
  tmp.cleanup()
})

const NEW_PROVIDER = {
  id: 'openai-main',
  label: 'OpenAI',
  kind: 'openai',
  baseUrl: 'https://api.openai.com/v1',
  model: 'gpt-4o-mini',
  structuredOutput: 'schema',
}

describe('GET /api/config', () => {
  it('返回当前 provider、provider 列表和语气表', async () => {
    const res = await request(app).get('/api/config')
    expect(res.status).toBe(200)
    expect(res.body.activeProviderId).toBe(store.getActiveProviderId())
    expect(res.body.providers.length).toBeGreaterThan(0)
    expect(res.body.tones).toHaveLength(8)
    expect(res.body.capabilities).toHaveLength(6)
  })
})

describe('provider 管理', () => {
  it('GET /api/providers 不回显明文 key（key 在 server/.env 里）', async () => {
    process.env.TEST_PROVIDER_KEY = 'sk-secret'
    await request(app)
      .put('/api/providers')
      .send({ ...NEW_PROVIDER, apiKeyEnv: 'TEST_PROVIDER_KEY' })

    const res = await request(app).get('/api/providers')
    expect(res.status).toBe(200)
    expect(JSON.stringify(res.body)).not.toContain('sk-secret')
    const found = res.body.providers.find((p: { id: string }) => p.id === 'openai-main')
    expect(found.hasKey).toBe(true)

    delete process.env.TEST_PROVIDER_KEY
  })

  it('PUT /api/providers 带 apiKey → 400（key 已经搬进 .env 了）', async () => {
    const res = await request(app)
      .put('/api/providers')
      .send({ ...NEW_PROVIDER, apiKey: 'sk-secret' })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/\.env/)
  })

  it('PUT /api/providers 新建 → 自动带一套独立的默认提示词', async () => {
    const res = await request(app).put('/api/providers').send(NEW_PROVIDER)
    expect(res.status).toBe(200)

    const prompts = await request(app).get('/api/providers/openai-main/prompts')
    expect(prompts.status).toBe(200)
    expect(prompts.body.templates).toHaveLength(6)
    expect(prompts.body.tones).toHaveLength(8)

    // 和原 provider 的提示词内容相同，但互不影响
    const original = store.getActiveProviderId()
    expect(prompts.body.templates[0].system).toBe(
      store.getPrompt(original, 'translate').system,
    )
  })

  it('PUT /api/providers 缺字段 → 400', async () => {
    expect((await request(app).put('/api/providers').send({})).status).toBe(400)
    expect(
      (await request(app).put('/api/providers').send({ ...NEW_PROVIDER, id: '' }))
        .status,
    ).toBe(400)
    expect(
      (await request(app).put('/api/providers').send({ ...NEW_PROVIDER, kind: 'foo' }))
        .status,
    ).toBe(400)
    expect(
      (await request(app).put('/api/providers').send({ ...NEW_PROVIDER, model: '' }))
        .status,
    ).toBe(400)
  })

  it('POST /api/providers/active 切换生效 provider', async () => {
    await request(app).put('/api/providers').send(NEW_PROVIDER)
    const res = await request(app)
      .post('/api/providers/active')
      .send({ id: 'openai-main' })
    expect(res.status).toBe(200)
    expect(res.body.activeProviderId).toBe('openai-main')
    expect(store.getActiveProviderId()).toBe('openai-main')
  })

  it('POST /api/providers/active 切到不存在的 → 404', async () => {
    const res = await request(app).post('/api/providers/active').send({ id: 'nope' })
    expect(res.status).toBe(404)
  })

  it('DELETE 当前生效的 provider → 400', async () => {
    const res = await request(app).delete(
      `/api/providers/${store.getActiveProviderId()}`,
    )
    expect(res.status).toBe(400)
  })

  it('DELETE 非生效的 provider → 200，连带清掉它的提示词', async () => {
    await request(app).put('/api/providers').send(NEW_PROVIDER)
    const res = await request(app).delete('/api/providers/openai-main')
    expect(res.status).toBe(200)
    expect(store.getProvider('openai-main')).toBeUndefined()
  })

  it('POST /api/providers/:id/test 用该 provider 探活', async () => {
    const res = await request(app).post(
      `/api/providers/${store.getActiveProviderId()}/test`,
    )
    expect(res.status).toBe(200)
    expect(res.body.ok).toBe(true)
    expect(res.body.sample).toBe('OK')
  })

  it('GET /api/providers/:id/models 列出模型', async () => {
    const res = await request(app).get(
      `/api/providers/${store.getActiveProviderId()}/models`,
    )
    expect(res.status).toBe(200)
    expect(res.body.models).toEqual(['stub-model'])
  })
})

describe('提示词（按 provider）', () => {
  it('GET /api/providers/:id/prompts 返回模板 + 语气 + 变量表', async () => {
    const id = store.getActiveProviderId()
    const res = await request(app).get(`/api/providers/${id}/prompts`)
    expect(res.status).toBe(200)
    expect(res.body.templates).toHaveLength(6)
    expect(res.body.tones).toHaveLength(8)
    expect(res.body.variables.translate).toContain('text')
    expect(res.body.samples.translate.text).toBeTruthy()
  })

  it('保存提示词 → 版本 +1', async () => {
    const id = store.getActiveProviderId()
    const before = await request(app).get(`/api/providers/${id}/prompts/translate`)
    const template = before.body.template

    const res = await request(app)
      .put(`/api/providers/${id}/prompts/translate`)
      .send({ template: { ...template, system: 'NEW SYSTEM' }, note: 'hello' })

    expect(res.status).toBe(200)
    expect(res.body.version).toBe(1)
    expect(store.getPrompt(id, 'translate').system).toBe('NEW SYSTEM')
  })

  it('各 provider 的提示词互相独立', async () => {
    const id = store.getActiveProviderId()
    await request(app).put('/api/providers').send(NEW_PROVIDER)

    const tpl = await request(app).get(`/api/providers/${id}/prompts/translate`)
    await request(app)
      .put(`/api/providers/${id}/prompts/translate`)
      .send({ template: { ...tpl.body.template, system: 'ONLY OLLAMA' } })

    const other = await request(app).get('/api/providers/openai-main/prompts/translate')
    expect(other.body.template.system).not.toBe('ONLY OLLAMA')
  })

  it('花括号不配平 → 400', async () => {
    const id = store.getActiveProviderId()
    const res = await request(app)
      .put(`/api/providers/${id}/prompts/translate`)
      .send({ template: { system: '{{#a}}x', user: 'U', responseMode: 'text' } })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/配平/)
  })

  it('JSON 模式缺字段 → 400', async () => {
    const id = store.getActiveProviderId()
    const res = await request(app)
      .put(`/api/providers/${id}/prompts/translate`)
      .send({
        template: {
          system: 'S',
          user: 'U',
          responseMode: 'json',
          jsonSchema: { type: 'object', properties: { wrong: { type: 'string' } } },
        },
      })
    expect(res.status).toBe(400)
    expect(res.body.error).toMatch(/translation/)
  })

  it('未知的 capability → 400', async () => {
    const id = store.getActiveProviderId()
    const res = await request(app).get(`/api/providers/${id}/prompts/nope`)
    expect(res.status).toBe(400)
  })

  it('版本历史 + 回滚', async () => {
    const id = store.getActiveProviderId()
    const tpl = await request(app).get(`/api/providers/${id}/prompts/translate`)

    await request(app)
      .put(`/api/providers/${id}/prompts/translate`)
      .send({ template: { ...tpl.body.template, system: 'A' }, note: 'first' })
    await request(app)
      .put(`/api/providers/${id}/prompts/translate`)
      .send({ template: { ...tpl.body.template, system: 'B' }, note: 'second' })

    const versions = await request(app).get(
      `/api/providers/${id}/prompts/translate/versions`,
    )
    expect(versions.body.versions).toHaveLength(2)
    expect(versions.body.versions[0].note).toBe('first')

    const one = await request(app).get(
      `/api/providers/${id}/prompts/translate/versions/1`,
    )
    expect(one.body.template.system).toBe('A')

    const rolled = await request(app)
      .post(`/api/providers/${id}/prompts/translate/rollback`)
      .send({ version: 1 })
    expect(rolled.status).toBe(200)
    expect(rolled.body.version).toBe(3)
    expect(store.getPrompt(id, 'translate').system).toBe('A')
  })

  it('回滚到不存在的版本 → 404', async () => {
    const id = store.getActiveProviderId()
    const res = await request(app)
      .post(`/api/providers/${id}/prompts/translate/rollback`)
      .send({ version: 99 })
    expect(res.status).toBe(404)
  })

  it('试运行：返回渲染后的提示词 + 原始输出 + 解析结果', async () => {
    const id = store.getActiveProviderId()
    const res = await request(app)
      .post(`/api/providers/${id}/prompts/translate/dry-run`)
      .send({ vars: { text: '今天天气很好。' } })

    expect(res.status).toBe(200)
    expect(res.body.system).toContain('Chinese-to-English translator')
    expect(res.body.user).toContain('Chinese: 今天天气很好。')
    expect(res.body.raw).toBe('OK')
    expect(res.body.parsed).toBe('OK')
  })

  it('试运行可以用未保存的模板', async () => {
    const id = store.getActiveProviderId()
    const res = await request(app)
      .post(`/api/providers/${id}/prompts/translate/dry-run`)
      .send({
        template: { system: 'CUSTOM', user: 'U', responseMode: 'text' },
        vars: { text: 'x' },
      })
    expect(res.status).toBe(200)
    expect(res.body.system).toContain('CUSTOM')
    // 没保存 → store 里还是原来的
    expect(store.getPrompt(id, 'translate').system).not.toContain('CUSTOM')
  })

  it('试运行缺变量 → 400', async () => {
    const id = store.getActiveProviderId()
    const res = await request(app)
      .post(`/api/providers/${id}/prompts/translate/dry-run`)
      .send({ vars: {} })
    expect(res.status).toBe(400)
  })

  it('从别的 provider 复制提示词', async () => {
    const id = store.getActiveProviderId()
    await request(app).put('/api/providers').send(NEW_PROVIDER)

    const tpl = await request(app).get(`/api/providers/${id}/prompts/translate`)
    await request(app)
      .put(`/api/providers/${id}/prompts/translate`)
      .send({ template: { ...tpl.body.template, system: 'TUNED' } })

    const res = await request(app)
      .post('/api/providers/openai-main/prompts/copy')
      .send({ fromProviderId: id, capabilityId: 'translate' })
    expect(res.status).toBe(200)

    const copied = await request(app).get('/api/providers/openai-main/prompts/translate')
    expect(copied.body.template.system).toBe('TUNED')
  })
})

describe('语气表', () => {
  it('GET /api/providers/:id/tones', async () => {
    const id = store.getActiveProviderId()
    const res = await request(app).get(`/api/providers/${id}/tones`)
    expect(res.status).toBe(200)
    expect(res.body.tones).toHaveLength(8)
  })

  it('PUT 覆盖语气表', async () => {
    const id = store.getActiveProviderId()
    const tones = store.getTones(id).map((t) => ({ ...t }))
    tones[0] = { ...tones[0], guidance: 'CHANGED' }

    const res = await request(app)
      .put(`/api/providers/${id}/tones`)
      .send({ tones })
    expect(res.status).toBe(200)
    expect(store.getTones(id)[0].guidance).toBe('CHANGED')
  })

  it('PUT 缺语气 → 400', async () => {
    const id = store.getActiveProviderId()
    const res = await request(app)
      .put(`/api/providers/${id}/tones`)
      .send({ tones: [{ id: 'formal', label: '正式', guidance: 'x' }] })
    expect(res.status).toBe(400)
  })
})
