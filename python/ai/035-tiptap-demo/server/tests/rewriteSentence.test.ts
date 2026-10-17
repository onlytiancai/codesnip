import { describe, expect, it } from 'vitest'
import {
  buildRewriteSentencePrompt,
  buildRewriteSentenceSystem,
  isSentenceOperation,
  parseRewrittenSentence,
} from '../src/prompts/rewriteSentence.js'

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

describe('buildRewriteSentenceSystem', () => {
  it('包含操作说明', () => {
    const s = buildRewriteSentenceSystem('improve')
    expect(s).toContain('Operation: improve')
    expect(s).toMatch(/grammar|collocation/i)
  })

  it('rewrite 携带 tone 指令', () => {
    const s = buildRewriteSentenceSystem('rewrite', 'formal')
    expect(s).toContain('Write in a formal tone')
    expect(s).toContain('no contractions')
  })

  it('非 rewrite 操作忽略 tone', () => {
    expect(buildRewriteSentenceSystem('improve', 'formal')).not.toContain(
      'Write in a formal tone',
    )
  })

  it('每个操作的说法各不相同', () => {
    const ops = ['rewrite', 'improve', 'shorten', 'expand'] as const
    const lines = ops.map(
      (op) => buildRewriteSentenceSystem(op).split('\n').at(-1),
    )
    expect(new Set(lines).size).toBe(4)
  })

  it('强约束纯文本输出', () => {
    const s = buildRewriteSentenceSystem('rewrite')
    expect(s).toContain('Output ONLY the rewritten sentence')
    expect(s).not.toMatch(/JSON array/)
  })
})

describe('buildRewriteSentencePrompt', () => {
  it('包含原句', () => {
    expect(buildRewriteSentencePrompt('Education is important.')).toContain(
      'Sentence: Education is important.',
    )
  })

  it('带 sourceSentence 时出现中文原文', () => {
    const p = buildRewriteSentencePrompt(
      'Education is important.',
      'First sentence.',
      '教育很重要。',
    )
    expect(p).toContain('Chinese source: 教育很重要。')
    expect(p).toContain('Keep the meaning aligned with the Chinese')
  })

  it('不带中文原文时不提对齐要求', () => {
    expect(buildRewriteSentencePrompt('x.')).not.toMatch(/aligned/)
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
