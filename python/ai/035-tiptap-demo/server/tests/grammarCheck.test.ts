import { describe, expect, it } from 'vitest'
import { renderCapability } from './helpers.js'

describe('grammarCheck 提示词模板', () => {
  it('user 就是原句本身，不加任何格式包装', () => {
    const u = renderCapability('grammarCheck', {
      sentence: 'He go to school.',
    }).user
    expect(u).toBe('He go to school.')
  })

  it('首尾空白被裁掉', () => {
    const u = renderCapability('grammarCheck', {
      sentence: '  She went home.  ',
    }).user
    expect(u).toBe('She went home.')
  })

  it('要求用中文回答', () => {
    const s = renderCapability('grammarCheck', { sentence: 'x' }).system
    expect(s).toMatch(/Chinese|中文/)
  })

  it('规定无错误时怎么说', () => {
    const s = renderCapability('grammarCheck', { sentence: 'x' }).system
    expect(s).toContain('这句话没有语法或标点错误。')
  })

  it('要求指出错在哪、改成什么', () => {
    const s = renderCapability('grammarCheck', { sentence: 'x' }).system
    expect(s).toMatch(/wrong words/)
    expect(s).toMatch(/instead/)
  })

  it('禁止报风格偏好（实测小模型最爱乱报这个）', () => {
    const s = renderCapability('grammarCheck', { sentence: 'x' }).system
    expect(s).toMatch(/Do not comment on style, tone/)
  })

  it('不要求任何输出格式 —— 不给格式就少一个失败点', () => {
    const s = renderCapability('grammarCheck', { sentence: 'x' }).system
    expect(s).not.toContain('|')
    expect(s).not.toContain('{')
    expect(s).not.toMatch(/JSON/i)
    expect(s).not.toMatch(/^\s*\d+\./m)
  })
})
