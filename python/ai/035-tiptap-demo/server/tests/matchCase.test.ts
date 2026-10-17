import { describe, expect, it } from 'vitest'
import { matchCase } from '../src/utils/matchCase.js'

describe('matchCase', () => {
  it('转全大写 → 目标全大写', () => {
    expect(matchCase('GOOD', 'great')).toBe('GREAT')
    expect(matchCase('USA', 'PRC')).toBe('PRC')
  })

  it('首字母大写 → 目标首字母大写', () => {
    expect(matchCase('Good', 'great')).toBe('Great')
    expect(matchCase('Apple', 'pear')).toBe('Pear')
  })

  it('首字母小写 → 目标保持不变', () => {
    expect(matchCase('good', 'GREAT')).toBe('GREAT')
    expect(matchCase('apple', 'Pear')).toBe('Pear')
  })

  it('空字符串 / 空 target 边界', () => {
    expect(matchCase('', 'great')).toBe('great')
    expect(matchCase('good', '')).toBe('')
  })

  it('单字符 "I" 走首字母大写分支', () => {
    expect(matchCase('I', 'me')).toBe('Me')
  })
})