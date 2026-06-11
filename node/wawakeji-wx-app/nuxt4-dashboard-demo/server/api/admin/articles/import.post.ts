import TurndownService from 'turndown'
import { gfm } from 'turndown-plugin-gfm'
import { Readability } from '@mozilla/readability'
import { JSDOM } from 'jsdom'
import { extractFromHtml } from '@extractus/article-extractor'
import { createHash } from 'crypto'
import { mkdir, writeFile } from 'fs/promises'

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const { url } = body

  // Validate URL
  if (!url) {
    throw createError({
      statusCode: 400,
      message: 'URL is required'
    })
  }

  let targetUrl: URL
  try {
    targetUrl = new URL(url)
    if (!['http:', 'https:'].includes(targetUrl.protocol)) {
      throw new Error('Invalid protocol')
    }
  } catch {
    throw createError({
      statusCode: 400,
      message: 'Invalid URL format'
    })
  }

  try {
    // Fetch HTML
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
      }
    })

    if (!response.ok) {
      throw createError({
        statusCode: response.status,
        message: `Failed to fetch URL: ${response.status} ${response.statusText}`
      })
    }

    const html = await response.text()

    // Extract content using article-extractor first, then fallback to Readability
    let extractedContent: { title: string; content: string } | null = null

    // Try article-extractor first
    try {
      const article = await extractFromHtml(html, url)
      if (article && article.content) {
        extractedContent = {
          title: article.title || 'Untitled',
          content: article.content
        }
      }
    } catch (e) {
      console.warn('article-extractor failed, trying Readability:', e)
    }

    // Fallback to Readability
    if (!extractedContent) {
      try {
        const dom = new JSDOM(html)
        const reader = new Readability(dom.window.document)
        const result = reader.parse()

        if (result && result.content) {
          extractedContent = {
            title: result.title || 'Untitled',
            content: result.content
          }
        }
      } catch (e) {
        console.warn('Readability failed:', e)
      }
    }

    // Final fallback: manual extraction
    if (!extractedContent) {
      extractedContent = extractContentFallback(html)
    }

    // Convert to Markdown
    let markdown = convertToMarkdown(extractedContent.content)

    // Extract and download images from the final markdown
    const imageMap = await extractAndDownloadImagesFromMarkdown(markdown, targetUrl)

    // Replace image URLs in markdown with local URLs
    for (const [originalUrl, localUrl] of Object.entries(imageMap)) {
      markdown = markdown.split(originalUrl).join(localUrl)
    }

    return {
      title: extractedContent.title,
      markdown,
      url,
      images: Object.entries(imageMap).map(([originalUrl, localUrl]) => ({
        originalUrl,
        localUrl
      }))
    }
  } catch (error: any) {
    console.error('Import error:', error)
    throw createError({
      statusCode: 500,
      message: error.message || 'Failed to import article'
    })
  }
})

/**
 * Extract and download images from Markdown content
 */
async function extractAndDownloadImagesFromMarkdown(markdown: string, baseUrl: URL): Promise<Record<string, string>> {
  const imageMap: Record<string, string> = {}

  // Match markdown image syntax: ![alt](url)
  const mdImageRegex = /!\[([^\]]*)\]\(([^)]+)\)/g
  const matches = [...markdown.matchAll(mdImageRegex)]

  for (const match of matches) {
    const originalUrl = match[2]

    // Skip data URLs, local paths, and already processed URLs
    if (originalUrl.startsWith('data:') || originalUrl.startsWith('/') || imageMap[originalUrl]) continue

    try {
      // Resolve relative URLs
      const imageUrl = new URL(originalUrl, baseUrl)

      // Download image
      const localUrl = await downloadImage(imageUrl.href)
      if (localUrl) {
        imageMap[originalUrl] = localUrl
      }
    } catch (e) {
      console.warn(`Failed to process image ${originalUrl}:`, e)
    }
  }

  return imageMap
}

/**
 * Download an image and save to local storage
 */
async function downloadImage(imageUrl: string): Promise<string | null> {
  try {
    const response = await fetch(imageUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
      }
    })

    if (!response.ok) return null

    const contentType = response.headers.get('content-type') || ''
    if (!contentType.startsWith('image/')) return null

    // Determine extension
    const ext = contentType.split('/')[1]?.split(';')[0] || 'jpg'
    const buffer = await response.arrayBuffer()

    // Generate unique filename
    const hash = createHash('md5')
      .update(imageUrl)
      .update(Buffer.from(buffer).toString('base64').slice(0, 100))
      .digest('hex')
      .slice(0, 12)

    const filename = `${Date.now()}-${hash}.${ext}`
    const uploadDir = 'public/uploads/images'

    // Ensure directory exists
    await mkdir(uploadDir, { recursive: true })

    // Save file
    await writeFile(`${uploadDir}/${filename}`, Buffer.from(buffer))

    return `/uploads/images/${filename}`
  } catch (e) {
    console.warn(`Failed to download image ${imageUrl}:`, e)
    return null
  }
}

/**
 * Fallback content extraction
 */
function extractContentFallback(html: string): { title: string; content: string } {
  // Simple regex-based extraction
  const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i)
  const title = titleMatch ? titleMatch[1].trim() : 'Untitled'

  // Try to find main content
  const contentMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i)
  let content = contentMatch ? contentMatch[1] : html

  // Remove script, style tags
  content = content.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
  content = content.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
  content = content.replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, '')
  content = content.replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, '')
  content = content.replace(/<header[^>]*>[\s\S]*?<\/header>/gi, '')

  return { title, content }
}

/**
 * Convert HTML to Markdown
 */
function convertToMarkdown(html: string): string {
  const turndownService = new TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced',
    fence: '```',
    emDelimiter: '*',
    bulletListMarker: '-'
  })

  // Use GFM plugin
  turndownService.use(gfm)

  // Custom code block rule
  turndownService.addRule('codeBlock', {
    filter: ['pre'],
    replacement: function (content: string, node: any) {
      const code = node.querySelector?.('code')
      if (code) {
        const className = code.className || ''
        const match = className.match(/language-(\w+)/)
        const language = match ? match[1] : ''
        return `\n\`\`\`${language}\n${code.textContent?.trim() || content.trim()}\n\`\`\`\n`
      }
      return `\n\`\`\`\n${content.trim()}\n\`\`\`\n`
    }
  })

  // Handle br tags
  turndownService.addRule('br', {
    filter: 'br',
    replacement: function () {
      return '  \n'
    }
  })

  return turndownService.turndown(html)
}