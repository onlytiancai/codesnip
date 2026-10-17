import { describe, expect, it } from 'vitest'
import {
  isToneId,
  TONE_GUIDANCE,
  TONE_IDS,
  toneInstruction,
} from '../src/prompts/tones.js'
import {
  buildTranslateSystemPrompt,
  TRANSLATE_SYSTEM_PROMPT,
} from '../src/prompts/translate.js'

describe('TONE_IDS', () => {
  it('覆盖需求的 8 种语气', () => {
    expect([...TONE_IDS]).toEqual([
      'neutral',
      'formal',
      'casual',
      'professional',
      'friendly',
      'concise',
      'detailed',
      'persuasive',
    ])
  })
})

describe('isToneId', () => {
  it('接受枚举内的 id', () => {
    expect(isToneId('formal')).toBe(true)
    expect(isToneId('persuasive')).toBe(true)
  })

  it('拒绝枚举外的值', () => {
    expect(isToneId('sarcastic')).toBe(false)
    expect(isToneId('')).toBe(false)
    expect(isToneId(null)).toBe(false)
    expect(isToneId(3)).toBe(false)
  })

  it('每个 id 都有 guidance', () => {
    for (const id of TONE_IDS) {
      expect(TONE_GUIDANCE[id]).toBeTruthy()
    }
  })
})

describe('toneInstruction', () => {
  it('产出带 id 与描述的一行', () => {
    const line = toneInstruction('formal')
    expect(line).toContain('formal')
    expect(line).toContain(TONE_GUIDANCE.formal)
  })

  it('缺省 / 非法 → undefined', () => {
    expect(toneInstruction(undefined)).toBeUndefined()
    expect(toneInstruction('nope')).toBeUndefined()
  })
})

describe('buildTranslateSystemPrompt', () => {
  it('无语气时与基础 prompt 逐字节相同', () => {
    expect(buildTranslateSystemPrompt()).toBe(TRANSLATE_SYSTEM_PROMPT)
    expect(buildTranslateSystemPrompt(undefined)).toBe(TRANSLATE_SYSTEM_PROMPT)
  })

  it('带语气时追加 tone 行，且基础约束仍在', () => {
    const p = buildTranslateSystemPrompt('concise')
    expect(p).toContain(TRANSLATE_SYSTEM_PROMPT)
    expect(p).toContain('concise')
    expect(p).toContain(TONE_GUIDANCE.concise)
    expect(p.length).toBeGreaterThan(TRANSLATE_SYSTEM_PROMPT.length)
  })
})
