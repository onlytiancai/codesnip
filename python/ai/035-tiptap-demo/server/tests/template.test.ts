import { describe, expect, it } from 'vitest'
import { isBalanced, render } from '../src/config/template.js'

describe('render', () => {
  it('替换变量', () => {
    expect(render('Hello {{name}}!', { name: 'World' })).toBe('Hello World!')
  })

  it('缺失的变量 → 空字符串', () => {
    expect(render('a{{missing}}b', {})).toBe('ab')
  })

  it('非字符串变量转成字符串', () => {
    expect(render('{{n}}', { n: 42 })).toBe('42')
    expect(render('{{b}}', { b: true })).toBe('true')
  })

  it('点号路径', () => {
    expect(render('{{a.b}}', { a: { b: 'x' } })).toBe('x')
    expect(render('{{a.b.c}}', { a: { b: { c: 'y' } } })).toBe('y')
  })

  it('点号路径穿过非对象 → 空字符串', () => {
    expect(render('{{a.b}}', { a: 'str' })).toBe('')
  })

  describe('section', () => {
    it('真值时渲染', () => {
      expect(render('{{#x}}yes{{/x}}', { x: 'v' })).toBe('yes')
      expect(render('{{#x}}yes{{/x}}', { x: true })).toBe('yes')
    })

    it('假值时跳过', () => {
      expect(render('{{#x}}yes{{/x}}', { x: '' })).toBe('')
      expect(render('{{#x}}yes{{/x}}', { x: false })).toBe('')
      expect(render('{{#x}}yes{{/x}}', { x: undefined })).toBe('')
      expect(render('{{#x}}yes{{/x}}', {})).toBe('')
    })

    it('inverted 相反', () => {
      expect(render('{{^x}}no{{/x}}', { x: '' })).toBe('no')
      expect(render('{{^x}}no{{/x}}', { x: 'v' })).toBe('')
    })

    it('可以嵌套', () => {
      const tpl = '{{#a}}A{{#b}}B{{/b}}{{/a}}'
      expect(render(tpl, { a: '1', b: '1' })).toBe('AB')
      expect(render(tpl, { a: '1', b: '' })).toBe('A')
      expect(render(tpl, { a: '', b: '1' })).toBe('')
    })
  })

  describe('standalone 行处理', () => {
    it('标签独占一行时，那一行只剩标签', () => {
      const tpl = ['Rules:', '{{#x}}', '- one', '{{/x}}', '- two'].join('\n')
      expect(render(tpl, { x: '1' })).toBe('Rules:\n- one\n- two')
    })

    it('标签不独占一行时不做裁剪', () => {
      const tpl = 'before {{#x}}mid{{/x}} after'
      expect(render(tpl, { x: '1' })).toBe('before mid after')
    })

    it('标签前的缩进被去掉', () => {
      const tpl = ['a', '  {{#x}}', 'b', '  {{/x}}', 'c'].join('\n')
      expect(render(tpl, { x: '1' })).toBe('a\nb\nc')
    })

    it('假值 section 整块消失，不留空行', () => {
      const tpl = ['Rules:', '{{#x}}', '- one', '{{/x}}', 'End'].join('\n')
      expect(render(tpl, { x: '' })).toBe('Rules:\nEnd')
    })
  })

  it('注释被忽略', () => {
    expect(render('a{{! this is a comment }}b', {})).toBe('ab')
  })

  it('三花括号也能用', () => {
    expect(render('{{{name}}}', { name: 'x' })).toBe('x')
  })
})

describe('isBalanced', () => {
  it('配平的模板通过', () => {
    expect(isBalanced('plain text')).toBe(true)
    expect(isBalanced('{{a}}')).toBe(true)
    expect(isBalanced('{{#a}}x{{/a}}')).toBe(true)
    expect(isBalanced('{{#a}}{{#b}}x{{/b}}{{/a}}')).toBe(true)
    expect(isBalanced('{{! comment }}')).toBe(true)
  })

  it('section 开闭不配平 → 不通过', () => {
    expect(isBalanced('{{#a}}x')).toBe(false)
    expect(isBalanced('{{/a}}')).toBe(false)
  })

  it('落单的 {{ }} → 不通过', () => {
    expect(isBalanced('{{a')).toBe(false)
    expect(isBalanced('a}}')).toBe(false)
  })
})
