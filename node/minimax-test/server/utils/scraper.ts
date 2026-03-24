import Turndown from 'turndown'
import { extract } from '@extractus/article-extractor'

const turndownService = new Turndown({
  headingStyle: 'atx',
  codeBlockStyle: 'fenced',
  bulletListMarker: '-'
})

// Custom rule for images
turndownService.addRule('img', {
  filter: 'img',
  replacement: (content, node) => {
    const img = node as HTMLImageElement
    const alt = img.alt || ''
    const src = img.getAttribute('src') || ''
    if (!src) return ''
    return `![${alt}](${src})`
  }
})

// Clean text helper
function cleanText(text: string): string {
  return text
    .replace(/[\u200B-\u200D\uFEFF\u00A0]/g, ' ') // Remove invisible chars
    .replace(/\r\n/g, '\n')
    .replace(/\t/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/ +\n/g, '\n')
    .replace(/\n +/g, '\n')
    .trim()
}

export interface ScrapedContent {
  title: string
  content: string
  description: string
  images: string[]
}

export async function scrapeUrl(url: string): Promise<ScrapedContent> {
  try {
    // Use article-extractor to extract main content
    const result = await extract(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
      }
    })

    if (!result) {
      throw new Error('Failed to extract content from URL')
    }

    const title = cleanText(result.title || 'Untitled')

    const description = cleanText(result.description || '')

    // Extract images from the result
    const images: string[] = []
    if (result.images && Array.isArray(result.images)) {
      for (const img of result.images) {
        if (img && typeof img === 'string') {
          try {
            const absoluteUrl = new URL(img, url).toString()
            images.push(absoluteUrl)
          } catch {
            images.push(img)
          }
        }
      }
    }

    // Convert images in content to absolute URLs
    let content = result.content || ''

    // Extract images from HTML content
    const imgRegex = /<img[^>]+src=["']([^"']+)["']/gi
    let match
    while ((match = imgRegex.exec(content)) !== null) {
      try {
        const absoluteUrl = new URL(match[1], url).toString()
        if (!images.includes(absoluteUrl)) {
          images.push(absoluteUrl)
        }
      } catch {
        // ignore invalid URLs
      }
    }

    // Replace relative image URLs with absolute in content
    content = content.replace(/src=["']([^"']*?)["']/gi, (match, src) => {
      if (src.startsWith('data:') || src.startsWith('blob:') || src.startsWith('http')) {
        return match
      }
      try {
        const absoluteUrl = new URL(src, url).toString()
        return `src="${absoluteUrl}"`
      } catch {
        return match
      }
    })

    // Convert to markdown
    let markdown = turndownService.turndown(content)

    // Clean up markdown
    markdown = cleanText(markdown)

    // Fix common issues
    markdown = markdown
      // Fix links with newlines inside brackets
      .replace(/\[([^\]]*?)\s*\n\s*([^\]]*?)\]/g, (match, p1, p2) => {
        return `[${p1}${p2}]`
      })
      // Fix newlines after opening bracket
      .replace(/\[\s*\n\s*/g, '[')
      // Fix newlines before closing bracket
      .replace(/\s*\n\s*\]/g, ']')
      // Fix empty lines in links
      .replace(/]\(\s*\n\s*/g, '](')
      .replace(/\s*\n\s*\)/g, ')')
      // Clean up multiple blank lines
      .replace(/\n{4,}/g, '\n\n\n')
      // Fix headers
      .replace(/^#{1,6}\s+.+$/gm, (line) => line.trim())
      // Fix code blocks
      .replace(/```\n{2,}/g, '```\n')
      // Remove leading/trailing whitespace from lines
      .split('\n')
      .map(line => line.trimEnd())
      .join('\n')

    return {
      title,
      content: markdown,
      description: description.substring(0, 500),
      images
    }
  } catch (error) {
    console.error('Scraping error:', error)
    throw error
  }
}
