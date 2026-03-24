import { describe, it, expect } from 'vitest'
import { markdownToHtml } from '../../server/utils/markdown'

describe('Markdown', () => {
  it('should convert basic markdown to HTML', () => {
    const markdown = '# Hello\n\nThis is a **bold** text.'
    const html = markdownToHtml(markdown)

    expect(html).toContain('<h1>')
    expect(html).toContain('Hello')
    expect(html).toContain('<strong>bold</strong>')
  })

  it('should convert links correctly', () => {
    const markdown = '[Click here](https://example.com)'
    const html = markdownToHtml(markdown)

    expect(html).toContain('<a href="https://example.com"')
    expect(html).toContain('Click here')
  })

  it('should convert images correctly', () => {
    const markdown = '![Alt text](/images/test.png)'
    const html = markdownToHtml(markdown)

    expect(html).toContain('<img')
    expect(html).toContain('src="/images/test.png"')
    expect(html).toContain('alt="Alt text"')
  })
})
