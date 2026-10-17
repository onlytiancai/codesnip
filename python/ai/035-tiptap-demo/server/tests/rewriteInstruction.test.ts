import { describe, expect, it } from 'vitest'
import {
  buildInstructionPrompt,
  INSTRUCTION_SYSTEM,
  parseInstructionResult,
} from '../src/prompts/rewriteInstruction.js'

describe('INSTRUCTION_SYSTEM', () => {
  it('强约束纯文本输出', () => {
    expect(INSTRUCTION_SYSTEM).toContain('Output ONLY the rewritten text')
    expect(INSTRUCTION_SYSTEM).not.toMatch(/JSON/)
  })
})

describe('buildInstructionPrompt', () => {
  it('原样带上指令与待改写文本', () => {
    const p = buildInstructionPrompt('He said hi.', 'Make it shorter.')
    expect(p).toContain('Instruction: Make it shorter.')
    expect(p).toContain('Text to rewrite: He said hi.')
  })

  it('context 明确标注不进输出', () => {
    const p = buildInstructionPrompt('hi.', 'Make it formal.', 'Some paragraph.')
    expect(p).toContain('Surrounding context')
    expect(p).toContain('do not include it in the output')
  })

  it('不带 context 时不出现该段', () => {
    expect(buildInstructionPrompt('hi.', 'formal')).not.toContain('Surrounding context')
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
