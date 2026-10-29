import { describe, expect, it } from 'vitest'
import {
  isSentenceOperation,
  parseRewrittenSentence,
} from '../src/prompts/rewriteSentence.js'
import { renderCapability } from './helpers.js'

describe('isSentenceOperation', () => {
  it('接受四个操作', () => {
    expect(isSentenceOperation('rewrite')).toBe(true)
    expect(isSentenceOperation('improve')).toBe(true)
    expect(isSentenceOperation('shorten')).toBe(true)
    expect(isSentenceOperation('expand')).toBe(true)
  })

  it('拒绝其它值', () => {
    expect(isSentenceOperation('polish')).toBe(false)
    expect(isSentenceOperation(null)).toBe(false)
    expect(isSentenceOperation(7)).toBe(false)
  })
})

describe('rewriteSentence 提示词模板', () => {
  it('包含操作说明', () => {
    const s = renderCapability('rewriteSentence', {
      sentence: 'x.',
      operation: 'improve',
    }).system
    expect(s).toContain('Operation: improve')
    expect(s).toMatch(/grammar|collocation/i)
  })

  it('rewrite 携带 tone 指令', () => {
    const s = renderCapability('rewriteSentence', {
      sentence: 'x.',
      operation: 'rewrite',
      tone: 'formal',
    }).system
    expect(s).toContain('Write in a formal tone')
    expect(s).toContain('no contractions')
  })

  it('非 rewrite 操作忽略 tone', () => {
    const s = renderCapability('rewriteSentence', {
      sentence: 'x.',
      operation: 'improve',
      tone: 'formal',
    }).system
    expect(s).not.toContain('Write in a formal tone')
  })

  it('每个操作的说法各不相同', () => {
    const ops = ['rewrite', 'improve', 'shorten', 'expand'] as const
    const lines = ops.map(
      (op) =>
        renderCapability('rewriteSentence', { sentence: 'x.', operation: op })
          .system.split('\n')
          .at(-1),
    )
    expect(new Set(lines).size).toBe(4)
  })

  it('强约束纯文本输出', () => {
    const s = renderCapability('rewriteSentence', {
      sentence: 'x.',
      operation: 'rewrite',
    }).system
    expect(s).toContain('Output ONLY the rewritten sentence')
    expect(s).not.toMatch(/JSON array/)
  })

  it('user 包含原句', () => {
    const u = renderCapability('rewriteSentence', {
      sentence: 'Education is important.',
      operation: 'rewrite',
    }).user
    expect(u).toContain('Sentence: Education is important.')
  })

  it('带 sourceSentence 时出现中文原文', () => {
    const u = renderCapability('rewriteSentence', {
      sentence: 'Education is important.',
      operation: 'rewrite',
      context: 'First sentence.',
      sourceSentence: '教育很重要。',
    }).user
    expect(u).toContain('Chinese source: 教育很重要。')
    expect(u).toContain('Keep the meaning aligned with the Chinese')
  })

  it('不带中文原文时不提对齐要求', () => {
    const u = renderCapability('rewriteSentence', {
      sentence: 'x.',
      operation: 'rewrite',
    }).user
    expect(u).not.toMatch(/aligned/)
  })
})

describe('parseRewrittenSentence', () => {
  it('纯文本直出', () => {
    expect(parseRewrittenSentence('Education matters.')).toBe('Education matters.')
  })

  it('剥围栏与前缀', () => {
    expect(parseRewrittenSentence('```\nEducation matters.\n```')).toBe(
      'Education matters.',
    )
    expect(parseRewrittenSentence('Rewritten: Education matters.')).toBe(
      'Education matters.',
    )
    expect(parseRewrittenSentence('改写：Education matters.')).toBe(
      'Education matters.',
    )
  })

  it('剥成对引号', () => {
    expect(parseRewrittenSentence('"Education matters."')).toBe('Education matters.')
  })

  it('空响应 → 抛错', () => {
    expect(() => parseRewrittenSentence('  ')).toThrow()
  })
})
