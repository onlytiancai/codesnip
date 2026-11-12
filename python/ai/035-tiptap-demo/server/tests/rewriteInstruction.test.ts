import { describe, expect, it } from 'vitest'
import { parseInstructionResult } from '../src/prompts/rewriteInstruction.js'
import { renderCapability } from './helpers.js'

describe('rewriteInstruction 提示词模板', () => {
  it('强约束纯文本输出', () => {
    const s = renderCapability('rewriteInstruction', {
      text: 'He said hi.',
      instruction: 'Make it shorter.',
    }).system
    expect(s).toContain('Output ONLY the rewritten text')
    expect(s).not.toMatch(/JSON/)
  })

  it('原样带上指令与待改写文本', () => {
    const u = renderCapability('rewriteInstruction', {
      text: 'He said hi.',
      instruction: 'Make it shorter.',
    }).user
    expect(u).toContain('Instruction: Make it shorter.')
    expect(u).toContain('Text to rewrite: He said hi.')
  })

  it('context 明确标注不进输出', () => {
    const u = renderCapability('rewriteInstruction', {
      text: 'hi.',
      instruction: 'Make it formal.',
      context: 'Some paragraph.',
    }).user
    expect(u).toContain('Surrounding context')
    expect(u).toContain('do not include it in the output')
  })

  it('不带 context 时不出现该段', () => {
    const u = renderCapability('rewriteInstruction', {
      text: 'hi.',
      instruction: 'formal',
    }).user
    expect(u).not.toContain('Surrounding context')
  })
})

describe('parseInstructionResult', () => {
  it('纯文本直出', () => {
    expect(parseInstructionResult('More academic prose.')).toBe('More academic prose.')
  })

  it('剥围栏与前缀', () => {
    expect(parseInstructionResult('```\nBetter.\n```')).toBe('Better.')
    expect(parseInstructionResult('Result: Better.')).toBe('Better.')
    expect(parseInstructionResult('结果：Better.')).toBe('Better.')
  })

  it('剥成对引号', () => {
    expect(parseInstructionResult('"Better."')).toBe('Better.')
  })

  it('空响应 → 抛错', () => {
    expect(() => parseInstructionResult('   ')).toThrow()
  })
})
