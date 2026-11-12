import { describe, expect, it } from 'vitest'
import {
  DEFAULT_TONES,
  isToneId,
  TONE_IDS,
  toneInstruction,
} from '../src/config/defaults.js'
import { renderCapability } from './helpers.js'

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
})

describe('DEFAULT_TONES', () => {
  it('每个 id 都有 guidance 和中文 label', () => {
    for (const id of TONE_IDS) {
      const tone = DEFAULT_TONES.find((t) => t.id === id)
      expect(tone).toBeDefined()
      expect(tone?.guidance).toBeTruthy()
      expect(tone?.label).toBeTruthy()
    }
  })

  it('id 集合与 TONE_IDS 完全一致', () => {
    expect(DEFAULT_TONES.map((t) => t.id)).toEqual([...TONE_IDS])
  })
})

describe('toneInstruction', () => {
  it('产出带 id 与描述的一行', () => {
    const line = toneInstruction(DEFAULT_TONES, 'formal')
    expect(line).toContain('formal')
    expect(line).toContain(
      DEFAULT_TONES.find((t) => t.id === 'formal')?.guidance ?? '',
    )
  })

  it('缺省 / 非法 → undefined', () => {
    expect(toneInstruction(DEFAULT_TONES, undefined)).toBeUndefined()
    expect(toneInstruction(DEFAULT_TONES, 'nope')).toBeUndefined()
  })

  it('语气表里没有这个 id → undefined', () => {
    expect(toneInstruction([], 'formal')).toBeUndefined()
  })
})

describe('语气注入到提示词', () => {
  it('翻译：无语气时与基础约束逐字节相同', () => {
    const plain = renderCapability('translate', { text: '你好' }).system
    expect(plain).not.toMatch(/tone/)

    const withTone = renderCapability('translate', {
      text: '你好',
      tone: 'concise',
    }).system
    expect(withTone).toContain(plain)
    expect(withTone).toContain('concise')
    expect(withTone).toContain(
      DEFAULT_TONES.find((t) => t.id === 'concise')?.guidance ?? '',
    )
    expect(withTone.length).toBeGreaterThan(plain.length)
  })
})
