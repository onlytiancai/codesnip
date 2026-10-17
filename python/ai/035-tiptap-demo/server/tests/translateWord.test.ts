import { describe, expect, it } from 'vitest'
import {
  buildTranslatePrompt,
  parseTranslationResult,
} from '../src/prompts/translate.js'

describe('buildTranslatePrompt', () => {
  it('包含原文', () => {
    const p = buildTranslatePrompt('我昨天去公园。')
    expect(p).toContain('我昨天去公园。')
  })

  it('强调 JSON 输出结构', () => {
    const p = buildTranslatePrompt('你好')
    expect(p).toMatch(/JSON/)
    expect(p).toMatch(/ONLY/)
    expect(p).toContain('translation')
    expect(p).toContain('sentences')
    expect(p).toContain('zh')
    expect(p).toContain('en')
  })

  it('强调逐句对齐', () => {
    const p = buildTranslatePrompt('你好')
    expect(p).toMatch(/sentence boundaries|aligned/i)
  })
})

describe('parseTranslationResult', () => {
  it('直接解析 JSON', () => {
    const raw = JSON.stringify({
      translation: 'I went to the park yesterday.',
      sentences: [{ zh: '我昨天去公园。', en: 'I went to the park yesterday.' }],
    })
    expect(parseTranslationResult(raw)).toEqual({
      translation: 'I went to the park yesterday.',
      sentences: [{ zh: '我昨天去公园。', en: 'I went to the park yesterday.' }],
    })
  })

  it('剥 ```json``` 围栏', () => {
    const raw =
      '```json\n' +
      JSON.stringify({
        translation: 'Hello.',
        sentences: [{ zh: '你好。', en: 'Hello.' }],
      }) +
      '\n```'
    expect(parseTranslationResult(raw)).toEqual({
      translation: 'Hello.',
      sentences: [{ zh: '你好。', en: 'Hello.' }],
    })
  })

  it('剥 ``` ``` 围栏（无语言标签）', () => {
    const raw =
      '```\n' +
      JSON.stringify({
        translation: 'Hello.',
        sentences: [{ zh: '你好。', en: 'Hello.' }],
      }) +
      '\n```'
    expect(parseTranslationResult(raw).translation).toBe('Hello.')
  })

  it('从混合文本里抠出 {...}', () => {
    const raw =
      'Here is the result:\n' +
      JSON.stringify({
        translation: 'Hello.',
        sentences: [{ zh: '你好。', en: 'Hello.' }],
      }) +
      '\nDone.'
    expect(parseTranslationResult(raw).translation).toBe('Hello.')
  })

  it('多条句子对齐', () => {
    const raw = JSON.stringify({
      translation: 'I went. And saw.',
      sentences: [
        { zh: '我去了。', en: 'I went.' },
        { zh: '并且看到了。', en: 'And saw.' },
      ],
    })
    expect(parseTranslationResult(raw).sentences).toHaveLength(2)
  })

  it('过滤非字符串 zh/en 元素', () => {
    const raw = JSON.stringify({
      translation: 'Hello.',
      sentences: [
        { zh: '你好。', en: 'Hello.' },
        { zh: 123, en: 'X' },
        { zh: 'Y', en: null },
        { zh: '', en: 'Z' },
        { zh: 'A', en: '' },
      ],
    })
    const result = parseTranslationResult(raw)
    expect(result.sentences).toEqual([{ zh: '你好。', en: 'Hello.' }])
  })

  it('去重（同样的 zh+en 重复出现只留一次）', () => {
    const raw = JSON.stringify({
      translation: 'Hello.',
      sentences: [
        { zh: '你好。', en: 'Hello.' },
        { zh: '你好。', en: 'Hello.' },
      ],
    })
    expect(parseTranslationResult(raw).sentences).toHaveLength(1)
  })

  it('空响应抛错', () => {
    expect(() => parseTranslationResult('')).toThrow()
    expect(() => parseTranslationResult('   ')).toThrow()
  })

  it('非 JSON 抛错', () => {
    expect(() => parseTranslationResult('not a json object')).toThrow()
  })

  it('顶层不是 object 抛错', () => {
    expect(() => parseTranslationResult('[1,2,3]')).toThrow()
  })

  it('缺 translation 字段抛错', () => {
    const raw = JSON.stringify({ sentences: [{ zh: '你好。', en: 'Hello.' }] })
    expect(() => parseTranslationResult(raw)).toThrow(/translation/)
  })

  it('translation 为空字符串抛错', () => {
    const raw = JSON.stringify({ translation: '   ', sentences: [{ zh: 'a', en: 'b' }] })
    expect(() => parseTranslationResult(raw)).toThrow(/translation/)
  })

  it('缺 sentences 数组抛错', () => {
    const raw = JSON.stringify({ translation: 'Hi' })
    expect(() => parseTranslationResult(raw)).toThrow(/sentences/)
  })

  it('sentences 全是非法元素 → 抛错', () => {
    const raw = JSON.stringify({
      translation: 'X',
      sentences: [{ zh: 1, en: 2 }, 'not object'],
    })
    expect(() => parseTranslationResult(raw)).toThrow(/no valid sentences/)
  })
})