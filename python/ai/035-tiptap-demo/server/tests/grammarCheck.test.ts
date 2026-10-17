import { describe, expect, it } from 'vitest'
import {
  buildGrammarCheckPrompt,
  GRAMMAR_SYSTEM,
} from '../src/prompts/grammarCheck.js'

describe('buildGrammarCheckPrompt', () => {
  it('就是原句本身，不加任何格式包装', () => {
    expect(buildGrammarCheckPrompt('He go to school.')).toBe('He go to school.')
  })

  it('首尾空白被裁掉', () => {
    expect(buildGrammarCheckPrompt('  She went home.  ')).toBe('She went home.')
  })
})

describe('GRAMMAR_SYSTEM', () => {
  it('要求用中文回答', () => {
    expect(GRAMMAR_SYSTEM).toMatch(/Chinese|中文/)
  })

  it('规定无错误时怎么说', () => {
    expect(GRAMMAR_SYSTEM).toContain('这句话没有语法或标点错误。')
  })

  it('要求指出错在哪、改成什么', () => {
    expect(GRAMMAR_SYSTEM).toMatch(/wrong words/)
    expect(GRAMMAR_SYSTEM).toMatch(/instead/)
  })

  it('禁止报风格偏好（实测小模型最爱乱报这个）', () => {
    expect(GRAMMAR_SYSTEM).toMatch(/Do not comment on style, tone/)
  })

  it('不要求任何输出格式 —— 不给格式就少一个失败点', () => {
    expect(GRAMMAR_SYSTEM).not.toContain('|')
    expect(GRAMMAR_SYSTEM).not.toContain('{')
    expect(GRAMMAR_SYSTEM).not.toMatch(/JSON/i)
    expect(GRAMMAR_SYSTEM).not.toMatch(/^\s*\d+\./m)
  })
})
