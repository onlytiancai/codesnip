import { describe, expect, it } from 'vitest'
import {
  buildContext,
  extractOutput,
  parseJsonLoose,
  renderPrompt,
} from '../src/config/capabilities.js'
import { DEFAULT_TONES } from '../src/config/defaults.js'
import { defaultPromptSet } from '../src/config/defaults.js'

const set = defaultPromptSet()

describe('extractOutput · text 模式', () => {
  it('translate：剥围栏和前缀', () => {
    expect(extractOutput('translate', 'I went home.', 'text')).toBe('I went home.')
    expect(extractOutput('translate', '```\nI went home.\n```', 'text')).toBe(
      'I went home.',
    )
  })

  it('rewriteSelection：解析 JSON 数组', () => {
    expect(
      extractOutput('rewriteSelection', '["great","promising"]', 'text'),
    ).toEqual(['great', 'promising'])
  })

  it('rewriteSelection：遵守 maxCandidates', () => {
    expect(
      extractOutput('rewriteSelection', '["a","b","c"]', 'text', {
        maxCandidates: 2,
      }),
    ).toEqual(['a', 'b'])
  })

  it('grammarCheck：原样透传，不做任何清洗', () => {
    const raw = '「go」应该改成「goes」。'
    expect(extractOutput('grammarCheck', raw, 'text')).toBe(raw)
    // 围栏也不剥 —— 老行为就是这样
    expect(extractOutput('grammarCheck', '```x```', 'text')).toBe('```x```')
  })

  it('rewriteInstruction：剥围栏', () => {
    expect(extractOutput('rewriteInstruction', '```\nBetter.\n```', 'text')).toBe(
      'Better.',
    )
  })
})

describe('extractOutput · json 模式', () => {
  it('translate：取 translation 字段', () => {
    expect(
      extractOutput('translate', '{"translation":"Hi."}', 'json'),
    ).toBe('Hi.')
  })

  it('rewriteSelection：取 candidates 数组', () => {
    expect(
      extractOutput('rewriteSelection', '{"candidates":["a","b"]}', 'json'),
    ).toEqual(['a', 'b'])
  })

  it('rewriteSentence / grammarCheck / rewriteInstruction：取 text 字段', () => {
    for (const cap of ['rewriteSentence', 'grammarCheck', 'rewriteInstruction'] as const) {
      expect(extractOutput(cap, '{"text":"x"}', 'json')).toBe('x')
    }
  })

  it('能从带围栏 / 带前后废话的输出里解出 JSON', () => {
    expect(
      extractOutput('translate', '```json\n{"translation":"Hi."}\n```', 'json'),
    ).toBe('Hi.')
    expect(
      extractOutput('translate', 'Here you go: {"translation":"Hi."}', 'json'),
    ).toBe('Hi.')
  })

  it('缺字段 → 报错', () => {
    expect(() => extractOutput('translate', '{"foo":"x"}', 'json')).toThrow(
      /missing the "translation" field/,
    )
  })

  it('字段类型不对 → 报错', () => {
    expect(() =>
      extractOutput('translate', '{"translation":123}', 'json'),
    ).toThrow(/must be a string/)
    expect(() =>
      extractOutput('rewriteSelection', '{"candidates":"x"}', 'json'),
    ).toThrow(/must be an array/)
  })

  it('不是 JSON → 报错', () => {
    expect(() => extractOutput('translate', 'sorry', 'json')).toThrow(
      /cannot parse JSON/,
    )
  })
})

describe('parseJsonLoose', () => {
  it('直接解', () => {
    expect(parseJsonLoose('{"a":1}')).toEqual({ a: 1 })
  })

  it('剥围栏', () => {
    expect(parseJsonLoose('```json\n{"a":1}\n```')).toEqual({ a: 1 })
  })

  it('抓第一个 {...}', () => {
    expect(parseJsonLoose('noise {"a":1} noise')).toEqual({ a: 1 })
  })

  it('空响应 → 抛错', () => {
    expect(() => parseJsonLoose('')).toThrow(/empty/)
  })
})

describe('buildContext', () => {
  it('rewriteSentence：只有 rewrite 才注入语气', () => {
    const rewrite = buildContext(
      'rewriteSentence',
      { sentence: 'x', operation: 'rewrite', tone: 'formal' },
      DEFAULT_TONES,
    )
    expect(rewrite.tone).toContain('Write in a formal tone')

    const improve = buildContext(
      'rewriteSentence',
      { sentence: 'x', operation: 'improve', tone: 'formal' },
      DEFAULT_TONES,
    )
    expect(improve.tone).toBeUndefined()
  })

  it('hasSource 由中文原句决定', () => {
    const withSrc = buildContext(
      'rewriteSelection',
      { text: 'x', kind: 'word', sourceSentence: '中文' },
      DEFAULT_TONES,
    )
    expect(withSrc.hasSource).toBe(true)

    const without = buildContext(
      'rewriteSelection',
      { text: 'x', kind: 'word' },
      DEFAULT_TONES,
    )
    expect(without.hasSource).toBe(false)
  })

  it('kind 决定 kindWord / kindPhrase', () => {
    const word = buildContext('rewriteSelection', { text: 'x', kind: 'word' }, DEFAULT_TONES)
    expect(word.kindWord).toBe(true)
    expect(word.kindPhrase).toBe(false)

    const phrase = buildContext('rewriteSelection', { text: 'x', kind: 'phrase' }, DEFAULT_TONES)
    expect(phrase.kindWord).toBe(false)
    expect(phrase.kindPhrase).toBe(true)
  })

  it('context / sourceSentence 会 trim', () => {
    const ctx = buildContext(
      'rewriteInstruction',
      { text: 'x', instruction: 'do it', context: '  para  ' },
      DEFAULT_TONES,
    )
    expect(ctx.context).toBe('para')
  })
})

describe('renderPrompt', () => {
  it('尾部空白被去掉', () => {
    const out = renderPrompt(
      {
        id: 'x',
        label: 'x',
        system: 'S\n',
        user: 'U\n\n',
        responseMode: 'text',
      },
      {},
    )
    expect(out.system).toBe('S')
    expect(out.user).toBe('U')
  })

  it('默认模板都能渲染出非空内容', () => {
    for (const [id, template] of Object.entries(set.templates)) {
      const out = renderPrompt(template, { text: 'x', sentence: 'x' })
      expect(out.system.length, id).toBeGreaterThan(0)
      expect(out.user.length, id).toBeGreaterThan(0)
    }
  })
})
