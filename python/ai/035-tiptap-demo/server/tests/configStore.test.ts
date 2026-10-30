import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { ConfigStore } from '../src/config/store.js'
import type { CapabilityId, ProviderConfig } from '../src/config/types.js'
import { buildTempStore, type TempStore } from './helpers.js'

let tmp: TempStore | undefined

function store(): ConfigStore {
  tmp = buildTempStore()
  return tmp.store
}

afterEach(() => {
  tmp?.cleanup()
  tmp = undefined
})

const NEW_PROVIDER: ProviderConfig = {
  id: 'openai-main',
  label: 'OpenAI',
  kind: 'openai',
  baseUrl: 'https://api.openai.com/v1',
  model: 'gpt-4o-mini',
  structuredOutput: 'schema',
}

describe('ConfigStore · 初始化', () => {
  it('首次运行播种默认 provider 和提示词集，并落盘', () => {
    const s = store()
    expect(s.listProviders()).toHaveLength(1)
    expect(s.getActiveProviderId()).toBe(s.listProviders()[0].id)
    expect(existsSync(join(tmp!.dir, 'config.json'))).toBe(true)
    expect(existsSync(join(tmp!.dir, 'history.json'))).toBe(true)
  })

  it('每个 provider 都有完整的提示词集', () => {
    const s = store()
    const id = s.getActiveProviderId()
    const set = s.getPromptSet(id)
    for (const cap of [
      'translate',
      'rewriteSelection',
      'rewriteSentence',
      'grammarCheck',
      'rewriteInstruction',
    ] as CapabilityId[]) {
      expect(set.templates[cap]).toBeDefined()
      expect(set.templates[cap].system).toBeTruthy()
    }
    expect(set.tones).toHaveLength(8)
  })

  it('config.json 损坏 → 回落默认配置，服务器仍能起来', () => {
    const s = store()
    const path = join(tmp!.dir, 'config.json')
    require('node:fs').writeFileSync(path, '{ not json', 'utf8')
    const s2 = new ConfigStore({ dataDir: tmp!.dir })
    expect(s2.listProviders().length).toBeGreaterThan(0)
    expect(s).toBeDefined()
  })
})

describe('ConfigStore · provider', () => {
  it('新增 provider 时用默认提示词集初始化它自己的一套', () => {
    const s = store()
    const original = s.getActiveProviderId()
    s.upsertProvider(NEW_PROVIDER)

    const set = s.getPromptSet('openai-main')
    expect(set.templates.translate).toBeDefined()
    expect(set.tones).toHaveLength(8)

    // 关键：两个 provider 的提示词互相独立
    const originalTemplate = s.getPrompt(original, 'translate')
    const newTemplate = s.getPrompt('openai-main', 'translate')
    expect(newTemplate.system).toBe(originalTemplate.system)
    expect(newTemplate).not.toBe(originalTemplate)
  })

  it('改一个 provider 的提示词不影响另一个', () => {
    const s = store()
    const original = s.getActiveProviderId()
    s.upsertProvider(NEW_PROVIDER)

    s.savePrompt('openai-main', 'translate', {
      ...s.getPrompt('openai-main', 'translate'),
      system: 'CHANGED',
    })

    expect(s.getPrompt('openai-main', 'translate').system).toBe('CHANGED')
    expect(s.getPrompt(original, 'translate').system).not.toBe('CHANGED')
  })

  it('切换生效 provider 后，getActiveProvider 跟着变', () => {
    const s = store()
    s.upsertProvider(NEW_PROVIDER)
    s.setActiveProvider('openai-main')
    expect(s.getActiveProvider().id).toBe('openai-main')
  })

  it('切换到不存在的 provider → 抛 NotFoundError', () => {
    const s = store()
    expect(() => s.setActiveProvider('nope')).toThrow(/不存在/)
  })

  it('不能删除当前生效的 provider', () => {
    const s = store()
    expect(() => s.deleteProvider(s.getActiveProviderId())).toThrow(
      /不能删除当前生效的 provider/,
    )
  })

  it('删除 provider 会连带清掉它的提示词集和历史', () => {
    const s = store()
    const original = s.getActiveProviderId()
    s.upsertProvider(NEW_PROVIDER)
    s.savePrompt('openai-main', 'translate', {
      ...s.getPrompt('openai-main', 'translate'),
      system: 'X',
    })

    s.deleteProvider('openai-main')

    expect(s.getProvider('openai-main')).toBeUndefined()
    expect(s.listVersions('openai-main', 'translate')).toHaveLength(0)
    expect(s.getActiveProviderId()).toBe(original)
  })
})

describe('ConfigStore · 版本历史', () => {
  it('保存提示词产生递增版本', () => {
    const s = store()
    const id = s.getActiveProviderId()
    const tpl = s.getPrompt(id, 'translate')

    const v1 = s.savePrompt(id, 'translate', { ...tpl, system: 'A' }, 'first')
    const v2 = s.savePrompt(id, 'translate', { ...tpl, system: 'B' }, 'second')

    expect(v1.version).toBe(1)
    expect(v2.version).toBe(2)
    expect(s.getPrompt(id, 'translate').system).toBe('B')
    expect(s.listVersions(id, 'translate')).toHaveLength(2)
  })

  it('回滚：复制旧内容成一个新版本，历史只增不减', () => {
    const s = store()
    const id = s.getActiveProviderId()
    const tpl = s.getPrompt(id, 'translate')
    s.savePrompt(id, 'translate', { ...tpl, system: 'A' })
    s.savePrompt(id, 'translate', { ...tpl, system: 'B' })

    const rolled = s.rollback(id, 'translate', 1)

    expect(rolled.version).toBe(3)
    expect(rolled.note).toMatch(/rollback to v1/)
    expect(s.getPrompt(id, 'translate').system).toBe('A')
    // 历史变成 3 条，旧的两条一字未动
    const versions = s.listVersions(id, 'translate')
    expect(versions).toHaveLength(3)
    expect(versions[0].template.system).toBe('A')
    expect(versions[1].template.system).toBe('B')
    expect(versions[2].template.system).toBe('A')
  })

  it('回滚到不存在的版本 → 抛错', () => {
    const s = store()
    const id = s.getActiveProviderId()
    expect(() => s.rollback(id, 'translate', 99)).toThrow(/不存在/)
  })

  it('历史落盘到 history.json，重开后还在', () => {
    const s = store()
    const id = s.getActiveProviderId()
    const tpl = s.getPrompt(id, 'translate')
    s.savePrompt(id, 'translate', { ...tpl, system: 'A' }, 'note here')

    const raw = JSON.parse(
      readFileSync(join(tmp!.dir, 'history.json'), 'utf8'),
    ) as Record<string, { version: number; note?: string }[]>
    expect(raw[`${id}/translate`]).toHaveLength(1)
    expect(raw[`${id}/translate`][0].note).toBe('note here')

    const s2 = new ConfigStore({ dataDir: tmp!.dir })
    expect(s2.listVersions(id, 'translate')).toHaveLength(1)
  })
})

describe('ConfigStore · 复制提示词', () => {
  it('从别的 provider 复制单个能力', () => {
    const s = store()
    const original = s.getActiveProviderId()
    s.upsertProvider(NEW_PROVIDER)
    s.savePrompt(original, 'translate', {
      ...s.getPrompt(original, 'translate'),
      system: 'TUNED',
    })

    s.copyPrompts(original, 'openai-main', 'translate')

    expect(s.getPrompt('openai-main', 'translate').system).toBe('TUNED')
    // 没被复制的那个能力不受影响
    expect(s.getPrompt('openai-main', 'rewriteSentence').system).not.toBe(
      'TUNED',
    )
  })

  it('从别的 provider 复制整套', () => {
    const s = store()
    const original = s.getActiveProviderId()
    s.upsertProvider(NEW_PROVIDER)

    s.copyPrompts(original, 'openai-main')

    expect(s.getPrompt('openai-main', 'grammarCheck').system).toBe(
      s.getPrompt(original, 'grammarCheck').system,
    )
  })
})

describe('ConfigStore · 密钥', () => {
  it('key 从 provider 指名的环境变量里读', () => {
    const s = store()
    process.env.TEST_PROVIDER_KEY = 'sk-from-env'
    s.upsertProvider({ ...NEW_PROVIDER, apiKeyEnv: 'TEST_PROVIDER_KEY' })

    expect(s.getApiKey('openai-main')).toBe('sk-from-env')
    expect(s.hasApiKey('openai-main')).toBe(true)

    delete process.env.TEST_PROVIDER_KEY
  })

  it('config.json 里永远拿不到明文 key', () => {
    const s = store()
    process.env.TEST_PROVIDER_KEY = 'sk-secret-value'
    s.upsertProvider({ ...NEW_PROVIDER, apiKeyEnv: 'TEST_PROVIDER_KEY' })

    const listed = s.listProviders()
    expect(JSON.stringify(listed)).not.toContain('sk-secret-value')

    const config = readFileSync(join(tmp!.dir, 'config.json'), 'utf8')
    expect(config).not.toContain('sk-secret-value')

    delete process.env.TEST_PROVIDER_KEY
  })

  it('没配 apiKeyEnv → 拿不到 key', () => {
    const s = store()
    s.upsertProvider(NEW_PROVIDER)
    expect(s.getApiKey('openai-main')).toBeUndefined()
    expect(s.hasApiKey('openai-main')).toBe(false)
  })

  it('环境变量被清掉 → 立刻拿不到 key（不用重启）', () => {
    const s = store()
    process.env.TEST_PROVIDER_KEY = 'sk-from-env'
    s.upsertProvider({ ...NEW_PROVIDER, apiKeyEnv: 'TEST_PROVIDER_KEY' })
    expect(s.getApiKey('openai-main')).toBe('sk-from-env')

    delete process.env.TEST_PROVIDER_KEY
    expect(s.getApiKey('openai-main')).toBeUndefined()
  })
})

describe('ConfigStore · 语气表', () => {
  it('按 provider 独立保存', () => {
    const s = store()
    const original = s.getActiveProviderId()
    s.upsertProvider(NEW_PROVIDER)

    const tones = s.getTones(original)
    tones[0] = { ...tones[0], guidance: 'CHANGED GUIDANCE' }
    s.setTones(original, tones)

    expect(s.getTones(original)[0].guidance).toBe('CHANGED GUIDANCE')
    expect(s.getTones('openai-main')[0].guidance).not.toBe('CHANGED GUIDANCE')
  })
})
