import { describe, expect, it } from 'vitest'
import { isSelectionKind, parseCandidates } from '../src/prompts/rewriteSelection.js'
import { renderCapability } from './helpers.js'

describe('isSelectionKind', () => {
  it('只接受 word / phrase', () => {
    expect(isSelectionKind('word')).toBe(true)
    expect(isSelectionKind('phrase')).toBe(true)
    expect(isSelectionKind('sentence')).toBe(false)
    expect(isSelectionKind(123)).toBe(false)
    expect(isSelectionKind(undefined)).toBe(false)
  })
})

describe('rewriteSelection 提示词模板', () => {
  it('word：要求同词性、禁形态变化', () => {
    const s = renderCapability('rewriteSelection', {
      text: 'good',
      kind: 'word',
    }).system
    expect(s).toContain('SAME part of speech')
    expect(s).toContain('morphological variants')
  })

  it('phrase：允许多词候选、强调搭配与语法', () => {
    const s = renderCapability('rewriteSelection', {
      text: 'very important',
      kind: 'phrase',
    }).system
    expect(s).toContain('multi-word phrases')
    expect(s).toContain('collocation')
    expect(s).toContain('surrounding text grammatical')
    expect(s).not.toContain('SAME part of speech')
  })

  it('两者都强约束 JSON 数组输出', () => {
    for (const kind of ['word', 'phrase'] as const) {
      const s = renderCapability('rewriteSelection', { text: 'x', kind }).system
      expect(s).toContain('Output ONLY a JSON array')
    }
  })

  it('大小写要求对多词候选同样生效', () => {
    const s = renderCapability('rewriteSelection', {
      text: 'x',
      kind: 'phrase',
    }).system
    expect(s).toContain('"Essential"')
  })

  it('user 包含选区', () => {
    const u = renderCapability('rewriteSelection', {
      text: 'very important',
      kind: 'phrase',
    }).user
    expect(u).toContain('Selection: very important')
  })

  it('带 context / sourceSentence 时分别出现', () => {
    const u = renderCapability('rewriteSelection', {
      text: 'important',
      kind: 'phrase',
      context: 'Education is very important for children.',
      sourceSentence: '教育对孩子很重要。',
    }).user
    expect(u).toContain('Context: Education is very important for children.')
    expect(u).toContain('Chinese source: 教育对孩子很重要。')
    expect(u).toContain('Use the Chinese to disambiguate')
  })

  it('不带中文原文时不提 disambiguate', () => {
    const u = renderCapability('rewriteSelection', {
      text: 'good',
      kind: 'word',
    }).user
    expect(u).not.toMatch(/disambiguate/)
  })
})

describe('parseCandidates', () => {
  it('干净 JSON 数组', () => {
    expect(parseCandidates('["great","promising"]')).toEqual(['great', 'promising'])
  })

  it('剥 ```json``` 围栏', () => {
    expect(parseCandidates('```json\n["great"]\n```')).toEqual(['great'])
    expect(parseCandidates('```\n["great"]\n```')).toEqual(['great'])
  })

  it('前后有零散文本时抓 [...] 子串', () => {
    expect(parseCandidates('Here are some: ["great","excellent"]')).toEqual([
      'great',
      'excellent',
    ])
  })

  it('大小写不敏感去重 + 截断到 max', () => {
    expect(parseCandidates('["Great","great","Good"]', 2)).toEqual(['Great', 'Good'])
  })

  it('丢非字符串项', () => {
    expect(parseCandidates('["great",42,null,"good"]')).toEqual(['great', 'good'])
  })

  it('全是非字符串 → 抛 no valid candidates', () => {
    expect(() => parseCandidates('[42,null]')).toThrow(/no valid/)
  })

  it('空响应 → 抛错', () => {
    expect(() => parseCandidates('')).toThrow(/empty/)
  })

  it('无法解析 → 抛错', () => {
    expect(() => parseCandidates('sorry I cannot')).toThrow(/cannot parse/)
  })
})
