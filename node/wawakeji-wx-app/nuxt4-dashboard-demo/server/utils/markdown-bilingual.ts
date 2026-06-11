import { translateText } from './translate'

export interface MarkdownBlock {
  type: 'paragraph' | 'heading' | 'list' | 'code' | 'blockquote' | 'table' | 'hr' | 'image'
  content: string
  level?: number // for headings
  language?: string // for code blocks
}

export interface BilingualBlock {
  en: string
  cn: string
  type: string
}

/**
 * Parse markdown into blocks
 */
export function parseMarkdownBlocks(markdown: string): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = []
  const lines = markdown.split('\n')
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // Code block
    if (line.startsWith('```')) {
      const language = line.slice(3).trim()
      const codeLines: string[] = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) {
        codeLines.push(lines[i])
        i++
      }
      i++ // skip closing ```
      blocks.push({
        type: 'code',
        content: codeLines.join('\n'),
        language
      })
      continue
    }

    // Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
      blocks.push({ type: 'hr', content: '' })
      i++
      continue
    }

    // Image (standalone)
    if (line.match(/^!\[.*?\]\(.*?\)$/)) {
      blocks.push({ type: 'image', content: line })
      i++
      continue
    }

    // Heading
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/)
    if (headingMatch) {
      blocks.push({
        type: 'heading',
        content: headingMatch[2],
        level: headingMatch[1].length
      })
      i++
      continue
    }

    // Table
    if (line.includes('|') && i + 1 < lines.length && lines[i + 1].match(/^\|?[\s-:|]+\|?$/)) {
      const tableLines: string[] = [line]
      i++
      tableLines.push(lines[i]) // separator line
      i++
      while (i < lines.length && lines[i].includes('|')) {
        tableLines.push(lines[i])
        i++
      }
      blocks.push({ type: 'table', content: tableLines.join('\n') })
      continue
    }

    // Blockquote
    if (line.startsWith('>')) {
      const quoteLines: string[] = []
      while (i < lines.length && lines[i].startsWith('>')) {
        quoteLines.push(lines[i].slice(1).trim())
        i++
      }
      blocks.push({
        type: 'blockquote',
        content: quoteLines.join('\n')
      })
      continue
    }

    // List
    if (line.match(/^(\s*)([-*+]|\d+\.)\s/)) {
      const listLines: string[] = []
      while (i < lines.length && lines[i].match(/^(\s*)([-*+]|\d+\.)\s/)) {
        listLines.push(lines[i])
        i++
      }
      blocks.push({ type: 'list', content: listLines.join('\n') })
      continue
    }

    // Empty line
    if (!line.trim()) {
      i++
      continue
    }

    // Paragraph
    const paragraphLines: string[] = []
    while (i < lines.length && lines[i].trim() && !isSpecialLine(lines[i])) {
      paragraphLines.push(lines[i])
      i++
    }
    if (paragraphLines.length > 0) {
      blocks.push({
        type: 'paragraph',
        content: paragraphLines.join('\n')
      })
    }
  }

  return blocks
}

function isSpecialLine(line: string): boolean {
  return (
    line.startsWith('```') ||
    line.startsWith('#') ||
    line.startsWith('>') ||
    line.match(/^(\s*)([-*+]|\d+\.)\s/) !== null ||
    line.includes('|') ||
    /^(-{3,}|\*{3,}|_{3,})$/.test(line.trim()) ||
    line.match(/^!\[.*?\]\(.*?\)$/) !== null
  )
}

/**
 * Translate markdown blocks and generate bilingual format
 */
export async function translateMarkdownToBilingual(
  markdown: string,
  onProgress?: (current: number, total: number) => void
): Promise<string> {
  const blocks = parseMarkdownBlocks(markdown)
  const results: string[] = []
  const translatableBlocks = blocks.filter(
    b => b.type !== 'code' && b.type !== 'hr' && b.type !== 'image' && b.content.trim()
  )

  let translatedCount = 0

  for (const block of blocks) {
    switch (block.type) {
      case 'code':
        results.push(`\`\`\`${block.language || ''}\n${block.content}\n\`\`\``)
        break

      case 'hr':
        results.push('---')
        break

      case 'image':
        results.push(block.content)
        break

      case 'heading':
        const translatedHeading = await translateText(block.content)
        translatedCount++
        onProgress?.(translatedCount, translatableBlocks.length)
        results.push(`${'#'.repeat(block.level!)} ${block.content}`)
        results.push('')
        results.push(`${'#'.repeat(block.level!)} ${translatedHeading.text}`)
        break

      case 'table':
        const translatedTable = await translateTable(block.content)
        translatedCount++
        onProgress?.(translatedCount, translatableBlocks.length)
        results.push(translatedTable)
        break

      case 'blockquote':
        const translatedQuote = await translateText(block.content)
        translatedCount++
        onProgress?.(translatedCount, translatableBlocks.length)
        results.push(':::bilingual')
        results.push(`> ${block.content}`)
        results.push('')
        results.push('---')
        results.push('')
        results.push(`> ${translatedQuote.text}`)
        results.push(':::')
        break

      case 'list':
        const translatedList = await translateList(block.content)
        translatedCount++
        onProgress?.(translatedCount, translatableBlocks.length)
        results.push(':::bilingual')
        results.push(block.content)
        results.push('')
        results.push('---')
        results.push('')
        results.push(translatedList)
        results.push(':::')
        break

      case 'paragraph':
      default:
        if (block.content.trim()) {
          const translated = await translateText(block.content)
          translatedCount++
          onProgress?.(translatedCount, translatableBlocks.length)
          results.push(':::bilingual')
          results.push(block.content)
          results.push('')
          results.push('---')
          results.push('')
          results.push(translated.text)
          results.push(':::')
        }
        break
    }

    results.push('') // Add spacing between blocks
  }

  return results.join('\n').trim()
}

async function translateTable(tableContent: string): Promise<string> {
  const lines = tableContent.split('\n')
  if (lines.length < 2) return tableContent

  const headerLine = lines[0]
  const separatorLine = lines[1]
  const dataLines = lines.slice(2)

  // Translate header
  const headerCells = headerLine.split('|').map(c => c.trim()).filter(Boolean)
  const translatedHeaders = await Promise.all(
    headerCells.map(cell => translateText(cell))
  )

  // Translate data rows
  const translatedDataLines = await Promise.all(
    dataLines.map(async (line) => {
      const cells = line.split('|').map(c => c.trim()).filter(Boolean)
      const translatedCells = await Promise.all(
        cells.map(cell => translateText(cell))
      )
      return `| ${translatedCells.map(r => r.text).join(' | ')} |`
    })
  )

  // Reconstruct table
  const result: string[] = []
  result.push(`| ${translatedHeaders.map(h => h.text).join(' | ')} |`)
  result.push(separatorLine)
  result.push(...translatedDataLines)

  return result.join('\n')
}

async function translateList(listContent: string): Promise<string> {
  const lines = listContent.split('\n')

  const translatedLines = await Promise.all(
    lines.map(async (line) => {
      const match = line.match(/^(\s*)([-*+]|\d+\.)\s+(.+)$/)
      if (match) {
        const indent = match[1]
        const marker = match[2]
        const content = match[3]
        const translated = await translateText(content)
        return `${indent}${marker} ${translated.text}`
      }
      return line
    })
  )

  return translatedLines.join('\n')
}

/**
 * Extract bilingual blocks from bilingual markdown
 */
export function extractBilingualBlocks(bilingualMarkdown: string): BilingualBlock[] {
  const blocks: BilingualBlock[] = []
  const regex = /:::bilingual\n([\s\S]*?)\n---\n([\s\S]*?)\n:::/g
  let match

  while ((match = regex.exec(bilingualMarkdown)) !== null) {
    const en = match[1].trim()
    const cn = match[2].trim()
    // Determine type based on content
    let type = 'paragraph'
    if (en.startsWith('>')) type = 'blockquote'
    else if (en.match(/^[-*+]\s/)) type = 'list'
    else if (en.match(/^\d+\.\s/)) type = 'list'

    blocks.push({ en, cn, type })
  }

  return blocks
}