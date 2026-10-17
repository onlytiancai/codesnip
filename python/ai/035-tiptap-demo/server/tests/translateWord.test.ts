import { describe, expect, it } from 'vitest'
import {
  buildTranslatePrompt,
  parseTranslation,
} from '../src/prompts/translate.js'

describe('buildTranslatePrompt', () => {
  it('包含原文', () => {
    const p = buildTranslatePrompt('我昨天去公园。')
    expect(p).toContain('Chinese: 我昨天去公园。')
    expect(p).toMatch(/English:/)
  })

  it('简洁无 JSON 约束', () => {
    const p = buildTranslatePrompt('你好')
    expect(p).not.toMatch(/JSON/)
  })
})

describe('parseTranslation', () => {
  it('纯英文一句话 → 直接返回', () => {
    expect(parseTranslation('I went to the park yesterday.')).toBe(
      'I went to the park yesterday.',
    )
  })

  it('trim 前后空白', () => {
    expect(parseTranslation('  Hello.  \n')).toBe('Hello.')
  })

  it('剥 ``` ``` 围栏', () => {
    expect(parseTranslation('```\nHello.\n```')).toBe('Hello.')
    expect(parseTranslation('```text\nHello.\n```')).toBe('Hello.')
  })

  it('剥 "English:" / "Translation:" 等前缀', () => {
    expect(parseTranslation('English: Hello.')).toBe('Hello.')
    expect(parseTranslation('Translation: Hello.')).toBe('Hello.')
    expect(parseTranslation('英文：你好。')).toBe('你好。')
  })

  it('剥首尾成对引号', () => {
    expect(parseTranslation('"Hello."')).toBe('Hello.')
    expect(parseTranslation("'Hello.'")).toBe('Hello.')
  })

  it('保留中间的标点', () => {
    expect(parseTranslation("It's a good idea.")).toBe("It's a good idea.")
    expect(parseTranslation('Hello, world!')).toBe('Hello, world!')
  })

  it('空响应抛错', () => {
    expect(() => parseTranslation('')).toThrow()
    expect(() => parseTranslation('   ')).toThrow()
  })

  it('只剥一层引号；不成对的引号保留', () => {
    expect(parseTranslation('"Hello.')).toBe('"Hello.')
    expect(parseTranslation('Hello.')).toBe('Hello.')
  })
})