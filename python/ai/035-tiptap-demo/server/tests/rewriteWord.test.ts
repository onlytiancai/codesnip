import { describe, expect, it } from 'vitest'
import { buildRewritePrompt, parseCandidates } from '../src/prompts/rewriteWord.js'

describe('buildRewritePrompt', () => {
  it('包含单词', () => {
    const p = buildRewritePrompt('good')
    expect(p).toContain('Word: good')
  })

  it('带 context 时附带', () => {
    const p = buildRewritePrompt('good', 'This is a good idea.')
    expect(p).toContain('Word: good')
    expect(p).toContain('Context: This is a good idea.')
  })

  it('context 是空白时不附带', () => {
    const p = buildRewritePrompt('good', '   ')
    expect(p).not.toContain('Context:')
  })

  it('prompt 强调 JSON-only', () => {
    const p = buildRewritePrompt('good')
    expect(p).toMatch(/JSON array/i)
    expect(p).toMatch(/ONLY/i)
  })
})

describe('parseCandidates', () => {
  it('直接解析 JSON 数组', () => {
    expect(parseCandidates('["great","promising","excellent"]')).toEqual([
      'great',
      'promising',
      'excellent',
    ])
  })

  it('剥 ```json``` 围栏', () => {
    expect(parseCandidates('```json\n["a","b","c"]\n```')).toEqual(['a', 'b', 'c'])
  })

  it('剥 ``` ``` 围栏（无语言标签）', () => {
    expect(parseCandidates('```\n["x","y"]\n```')).toEqual(['x', 'y'])
  })

  it('从混合文本里抠出 [...]', () => {
    expect(parseCandidates('Here you go: ["one","two","three"]!')).toEqual([
      'one',
      'two',
      'three',
    ])
  })

  it('trim 并跳过空字符串', () => {
    expect(parseCandidates('["  good  ","","great",""]')).toEqual(['good', 'great'])
  })

  it('大小写不敏感去重', () => {
    expect(parseCandidates('["Good","good","GOOD"]')).toEqual(['Good'])
  })

  it('限制最大数量', () => {
    expect(parseCandidates('["a","b","c","d","e","f"]', 3)).toEqual(['a', 'b', 'c'])
  })

  it('空响应抛错', () => {
    expect(() => parseCandidates('')).toThrow()
    expect(() => parseCandidates('   ')).toThrow()
  })

  it('非 JSON 抛错', () => {
    expect(() => parseCandidates('not a json')).toThrow()
  })

  it('顶层不是数组抛错', () => {
    expect(() => parseCandidates('{"a":1}')).toThrow()
  })

  it('全是空字符串 / 非字符串元素 → 抛错', () => {
    expect(() => parseCandidates('["", "", ""]')).toThrow()
    expect(() => parseCandidates('[1,2,3]')).toThrow()
  })
})