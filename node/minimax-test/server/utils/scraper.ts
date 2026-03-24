import Turndown from 'turndown'
import * as cheerio from 'cheerio'

const turndownService = new TurndownService()

function TurndownService() {
  const td = new Turndown({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced'
  })

  td.addRule('img', {
    filter: 'img',
    replacement: (content, node) => {
      const img = node as HTMLImageElement
      return `![${img.alt || ''}](${img.src})`
    }
  })

  return td
}

export interface ScrapedContent {
  title: string
  content: string
  description: string
  images: string[]
}

export async function scrapeUrl(url: string): Promise<ScrapedContent> {
  const response = await fetch(url, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch URL: ${response.statusText}`)
  }

  const html = await response.text()
  const $ = cheerio.load(html)

  $('script, style, nav, header, footer, aside, .ads, .advertisement, .sidebar').remove()

  const title = $('h1').first().text().trim() || $('title').text().trim() || 'Untitled'

  const description = $('meta[name="description"]').attr('content')
    || $('meta[property="og:description"]').attr('content')
    || ''

  const articleContent = $('article').first().html()
    || $('main').first().html()
    || $('[role="main"]').first().html()
    || $('body').first().html()
    || ''

  const images: string[] = []
  $('img').each((_, img) => {
    const src = $(img).attr('src')
    if (src && !src.startsWith('data:') && !src.startsWith('javascript:')) {
      images.push(src)
    }
  })

  const markdown = turndownService.turndown(articleContent)

  return {
    title,
    content: markdown,
    description: description.substring(0, 500),
    images
  }
}
