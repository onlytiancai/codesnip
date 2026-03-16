import TurndownService from 'turndown'
import { gfm } from 'turndown-plugin-gfm'
import { Readability } from '@mozilla/readability'
import { JSDOM } from 'jsdom'
import { extractFromHtml } from '@extractus/article-extractor'
import { createHash } from 'crypto'
import { mkdir, writeFile, unlink } from 'fs/promises'
import { existsSync } from 'fs'

// Stage progress percentages
export const STAGE_PROGRESS = {
  init: 0,
  fetch: 10,
  extract: 25,
  convert: 35,
  images_start: 40,
  done: 100
} as const

export type ImportStage = keyof typeof STAGE_PROGRESS | 'images'
export type ImportStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'

export interface ImportJobResult {
  title: string
  markdown: string
  url: string
  images: Array<{ originalUrl: string; localUrl: string }>
}

/**
 * Update import queue job status and progress
 */
export async function updateJobProgress(
  jobId: number,
  updates: {
    status?: ImportStatus
    stage?: ImportStage
    progress?: number
    totalImages?: number
    processedImages?: number
    result?: ImportJobResult
    error?: string
  }
) {
  await prisma.importQueue.update({
    where: { id: jobId },
    data: {
      ...updates,
      updatedAt: new Date()
    }
  })
}

/**
 * Fetch HTML from URL
 */
export async function fetchHtml(url: string): Promise<string> {
  const response = await fetch(url, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.5'
    }
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch URL: ${response.status} ${response.statusText}`)
  }

  return response.text()
}

/**
 * Extract content from HTML
 */
export function extractContent(html: string, url: string): { title: string; content: string } {
  let extractedContent: { title: string; content: string } | null = null

  // Try article-extractor first
  try {
    const article = extractFromHtml(html, url)
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

  return extractedContent
}

/**
 * Fallback content extraction
 */
function extractContentFallback(html: string): { title: string; content: string } {
  const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i)
  const title = titleMatch ? titleMatch[1].trim() : 'Untitled'

  const contentMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i)
  let content = contentMatch ? contentMatch[1] : html

  // Remove script, style, nav, footer, header tags
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
export function convertToMarkdown(html: string): string {
  const turndownService = new TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced',
    fence: '```',
    emDelimiter: '*',
    bulletListMarker: '-'
  })

  turndownService.use(gfm)

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

  turndownService.addRule('br', {
    filter: 'br',
    replacement: function () {
      return '  \n'
    }
  })

  return turndownService.turndown(html)
}

/**
 * Extract image URLs from Markdown content
 */
export function extractImageUrls(markdown: string): string[] {
  const mdImageRegex = /!\[([^\]]*)\]\(([^)]+)\)/g
  const matches = [...markdown.matchAll(mdImageRegex)]
  const urls = new Set<string>()

  for (const match of matches) {
    const url = match[2]
    // Skip data URLs and local paths
    if (!url.startsWith('data:') && !url.startsWith('/')) {
      urls.add(url)
    }
  }

  return Array.from(urls)
}

/**
 * Download an image and save to local storage
 */
export async function downloadImage(
  imageUrl: string,
  queueId: number,
  index: number
): Promise<{ localPath: string; localUrl: string } | null> {
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

    // Generate unique filename with queue ID prefix
    const hash = createHash('md5')
      .update(imageUrl)
      .update(Buffer.from(buffer).toString('base64').slice(0, 100))
      .digest('hex')
      .slice(0, 12)

    const filename = `${queueId}-${Date.now()}-${hash}.${ext}`
    const uploadDir = 'public/uploads/images'

    // Ensure directory exists
    await mkdir(uploadDir, { recursive: true })

    // Save file
    const localPath = `${uploadDir}/${filename}`
    await writeFile(localPath, Buffer.from(buffer))

    return {
      localPath,
      localUrl: `/uploads/images/${filename}`
    }
  } catch (e) {
    console.warn(`Failed to download image ${imageUrl}:`, e)
    return null
  }
}

/**
 * Delete imported images from filesystem
 */
export async function deleteImportImages(imagePaths: string[]): Promise<void> {
  for (const path of imagePaths) {
    try {
      if (existsSync(path)) {
        await unlink(path)
      }
    } catch (e) {
      console.warn(`Failed to delete image ${path}:`, e)
    }
  }
}

/**
 * Process an import job through all stages
 */
export async function processImportJob(jobId: number): Promise<void> {
  // Get job
  const job = await prisma.importQueue.findUnique({
    where: { id: jobId }
  })

  if (!job || job.status === 'cancelled') {
    return
  }

  // Mark as processing
  await updateJobProgress(jobId, { status: 'processing', stage: 'fetch', progress: STAGE_PROGRESS.fetch })

  try {
    const baseUrl = new URL(job.url)

    // Stage 1: Fetch HTML
    if (job.status === 'cancelled') return
    const html = await fetchHtml(job.url)
    await updateJobProgress(jobId, { stage: 'extract', progress: STAGE_PROGRESS.extract })

    // Stage 2: Extract content
    if (job.status === 'cancelled') return
    const extractedContent = extractContent(html, job.url)
    await updateJobProgress(jobId, { stage: 'convert', progress: STAGE_PROGRESS.convert })

    // Stage 3: Convert to Markdown
    if (job.status === 'cancelled') return
    let markdown = convertToMarkdown(extractedContent.content)
    await updateJobProgress(jobId, { stage: 'images', progress: STAGE_PROGRESS.images_start })

    // Stage 4: Extract and download images
    if (job.status === 'cancelled') return
    const imageUrls = extractImageUrls(markdown)
    const totalImages = imageUrls.length
    await updateJobProgress(jobId, { totalImages })

    const imageMap: Record<string, string> = {}
    const imageRecords: Array<{ originalUrl: string; localPath: string }> = []

    for (let i = 0; i < imageUrls.length; i++) {
      // Check for cancellation
      const currentJob = await prisma.importQueue.findUnique({ where: { id: jobId } })
      if (currentJob?.status === 'cancelled') return

      const originalUrl = imageUrls[i]

      // Resolve relative URLs
      let resolvedUrl: string
      try {
        resolvedUrl = new URL(originalUrl, baseUrl).href
      } catch {
        console.warn(`Invalid image URL: ${originalUrl}`)
        continue
      }

      // Create ImportImage record
      const importImage = await prisma.importImage.create({
        data: {
          queueId: jobId,
          originalUrl: resolvedUrl,
          status: 'downloading'
        }
      })

      try {
        const result = await downloadImage(resolvedUrl, jobId, i)

        if (result) {
          imageMap[originalUrl] = result.localUrl
          imageRecords.push({ originalUrl: resolvedUrl, localPath: result.localPath })

          await prisma.importImage.update({
            where: { id: importImage.id },
            data: { status: 'completed', localPath: result.localPath }
          })
        } else {
          await prisma.importImage.update({
            where: { id: importImage.id },
            data: { status: 'failed', error: 'Download failed' }
          })
        }
      } catch (e: any) {
        await prisma.importImage.update({
          where: { id: importImage.id },
          data: { status: 'failed', error: e.message }
        })
      }

      // Update progress
      const progress = Math.round(STAGE_PROGRESS.images_start + ((i + 1) / totalImages) * 60)
      await updateJobProgress(jobId, { processedImages: i + 1, progress })
    }

    // Replace image URLs in markdown with local URLs
    for (const [originalUrl, localUrl] of Object.entries(imageMap)) {
      markdown = markdown.split(originalUrl).join(localUrl)
    }

    // Stage 5: Complete
    const result: ImportJobResult = {
      title: extractedContent.title,
      markdown,
      url: job.url,
      images: Object.entries(imageMap).map(([originalUrl, localUrl]) => ({
        originalUrl,
        localUrl
      }))
    }

    await updateJobProgress(jobId, {
      status: 'completed',
      stage: 'done',
      progress: STAGE_PROGRESS.done,
      result
    })
  } catch (error: any) {
    console.error('Import job failed:', error)
    await updateJobProgress(jobId, {
      status: 'failed',
      error: error.message || 'Unknown error'
    })
  }
}