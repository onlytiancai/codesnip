import { translateText } from '../../../utils/translate'
import { parseMarkdownBlocks } from '../../../utils/markdown-bilingual'

interface TranslateRequest {
  markdown: string
  title?: string
}

interface BlockResult {
  original: string
  translated: string
  type: string
  level?: number
  language?: string
}

// Placeholder pattern for protecting inline code and links during translation
// Using a format that looks like a special token and won't be translated
const PLACEHOLDER_PREFIX = '[[CODE_'
const PLACEHOLDER_SUFFIX = ']]'
const INLINE_CODE_REGEX = /`([^`]+)`/g
const LINK_REGEX = /\[([^\]]*)\]\(([^)]+)\)/g
const CODE_FILENAME_REGEX = /([a-zA-Z0-9_\-./]+\.ts|[a-zA-Z0-9_\-./]+\.js|[a-zA-Z0-9_\-./]+\.vue|[a-zA-Z0-9_\-./]+\.json)/g

interface PlaceholderContext {
  store: Map<string, string>
  counter: number
}

/**
 * Create a new placeholder context for each translation call
 */
function createPlaceholderContext(): PlaceholderContext {
  return {
    store: new Map<string, string>(),
    counter: 0
  }
}

/**
 * Protect inline code and links by replacing them with placeholders
 * Returns the protected text and the placeholder context
 */
function protectSpecialContent(text: string, ctx: PlaceholderContext): string {
  let protectedText = text

  // Create a unique placeholder
  const createPlaceholder = () => {
    const id = ctx.counter++
    // Use a format that looks like a token and won't be translated
    return `${PLACEHOLDER_PREFIX}${id.toString(36).toUpperCase().padStart(3, '0')}${PLACEHOLDER_SUFFIX}`
  }

  // Protect links first (most complex pattern)
  protectedText = protectedText.replace(LINK_REGEX, (_, linkText, url) => {
    const placeholder = createPlaceholder()
    ctx.store.set(placeholder, `[${linkText}](${url})`)
    return placeholder
  })

  // Protect inline code
  protectedText = protectedText.replace(INLINE_CODE_REGEX, (_, code) => {
    const placeholder = createPlaceholder()
    ctx.store.set(placeholder, `\`${code}\``)
    return placeholder
  })

  // Protect common file names that might get translated
  protectedText = protectedText.replace(CODE_FILENAME_REGEX, (match) => {
    // Only protect if it looks like a filename (not already in backticks or link)
    const placeholder = createPlaceholder()
    ctx.store.set(placeholder, match)
    return placeholder
  })

  return protectedText
}

/**
 * Restore protected content after translation
 * Uses regex to handle potential spaces added by Google Translate
 */
function restoreSpecialContent(translated: string, ctx: PlaceholderContext): string {
  let result = translated

  console.log('[Translation Debug] Restoring content...')
  console.log('[Translation Debug] Placeholder store size:', ctx.store.size)
  console.log('[Translation Debug] Translated text sample:', translated.substring(0, 500))

  // Sort by placeholder ID (longest first) to avoid partial matches
  const sortedEntries = [...ctx.store.entries()].sort((a, b) => {
    // Extract numeric ID for proper sorting (e.g., "000", "001", "00A")
    const getNumId = (s: string) => {
      const m = s.match(/CODE_([A-Z0-9]+)/)
      return m ? m[1] : ''
    }
    return getNumId(b[0]).localeCompare(getNumId(a[0]))
  })

  for (const [placeholder, original] of sortedEntries) {
    // Extract the ID part (e.g., "000" from "[[CODE_000]]")
    const idMatch = placeholder.match(/CODE_([A-Z0-9]+)/)
    if (idMatch) {
      const id = idMatch[1]
      console.log(`[Translation Debug] Restoring placeholder CODE_${id} -> "${original.substring(0, 50)}..."`)

      // Create a flexible regex that handles potential spaces added by translation
      // Matches: [[CODE_000]], [[ CODE_000 ]], [[CODE_ 000]], etc.
      const flexibleRegex = new RegExp(
        `\\[\\[\\s*CODE\\s*_\\s*${id}\\s*\\]\\]`,
        'g'
      )
      const beforeReplace = result
      result = result.replace(flexibleRegex, original)

      // Check if replacement happened
      if (beforeReplace === result) {
        console.log(`[Translation Debug] WARNING: No match found for CODE_${id}`)
        // Try to find the placeholder in result
        if (result.includes(`[[CODE_${id}]]`)) {
          console.log(`[Translation Debug] Placeholder [[CODE_${id}]] found as literal in result`)
        } else if (result.includes(`CODE_${id}`)) {
          console.log(`[Translation Debug] CODE_${id} found in result (without brackets)`)
        }
      }
    } else {
      // Fallback to direct replacement
      console.log(`[Translation Debug] Using fallback for: ${placeholder}`)
      result = result.split(placeholder).join(original)
    }
  }

  console.log('[Translation Debug] Final result sample:', result.substring(0, 500))
  return result
}

/**
 * Translate text while protecting inline code and links
 */
async function translateWithProtection(text: string): Promise<string> {
  const ctx = createPlaceholderContext()
  const protectedText = protectSpecialContent(text, ctx)
  console.log('[Translation Debug] Protected text sample:', protectedText.substring(0, 300))
  const result = await translateText(protectedText)
  console.log('[Translation Debug] Translated text sample:', result.text.substring(0, 300))
  return restoreSpecialContent(result.text, ctx)
}

/**
 * Translate heading content, handling links specially:
 * - Extract link text and translate it
 * - Keep the URL unchanged
 * - Reconstruct the link after translation
 */
async function translateHeadingWithLinks(text: string): Promise<string> {
  // Check if the heading contains links
  const linkMatches = [...text.matchAll(LINK_REGEX)]

  if (linkMatches.length === 0) {
    // No links, use normal translation
    return translateWithProtection(text)
  }

  // Extract all link texts and translate them
  const linkTexts = linkMatches.map(m => m[1])
  const translatedTexts = await Promise.all(
    linkTexts.map(t => translateWithProtection(t))
  )

  // Replace link texts in original, keeping URLs
  let result = text
  let offset = 0
  linkMatches.forEach((match, i) => {
    const fullMatch = match[0]
    const linkText = match[1]
    const url = match[2]
    const translatedText = translatedTexts[i]

    // Find position in current result (accounting for previous replacements)
    const startPos = result.indexOf(fullMatch, offset)
    if (startPos !== -1) {
      const newLink = `[${translatedText}](${url})`
      result = result.slice(0, startPos) + newLink + result.slice(startPos + fullMatch.length)
      offset = startPos + newLink.length
    }
  })

  return result
}

export default defineEventHandler(async (event) => {
  const body = await readBody<TranslateRequest>(event)

  if (!body?.markdown) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Markdown content is required'
    })
  }

  const { markdown, title } = body
  const blocks = parseMarkdownBlocks(markdown)
  const results: BlockResult[] = []

  // Translate title if provided
  let translatedTitle = ''
  if (title) {
    try {
      const titleResult = await translateText(title)
      translatedTitle = titleResult.text
    } catch (error) {
      console.error('Failed to translate title:', error)
      translatedTitle = title // Fallback to original
    }
  }

  // Translate each block
  for (const block of blocks) {
    switch (block.type) {
      case 'code':
        // Don't translate code blocks
        results.push({
          original: block.content,
          translated: block.content,
          type: block.type,
          language: block.language
        })
        break

      case 'hr':
        results.push({
          original: '',
          translated: '',
          type: block.type
        })
        break

      case 'image':
        results.push({
          original: block.content,
          translated: block.content,
          type: block.type
        })
        break

      case 'heading':
        try {
          const translated = await translateHeadingWithLinks(block.content)
          results.push({
            original: block.content,
            translated,
            type: block.type,
            level: block.level
          })
        } catch (error) {
          console.error('Failed to translate heading:', error)
          results.push({
            original: block.content,
            translated: block.content,
            type: block.type,
            level: block.level
          })
        }
        break

      case 'table':
        try {
          const translatedTable = await translateTable(block.content)
          results.push({
            original: block.content,
            translated: translatedTable,
            type: block.type
          })
        } catch (error) {
          console.error('Failed to translate table:', error)
          results.push({
            original: block.content,
            translated: block.content,
            type: block.type
          })
        }
        break

      case 'blockquote':
        try {
          const translated = await translateWithProtection(block.content)
          results.push({
            original: block.content,
            translated,
            type: block.type
          })
        } catch (error) {
          console.error('Failed to translate blockquote:', error)
          results.push({
            original: block.content,
            translated: block.content,
            type: block.type
          })
        }
        break

      case 'list':
        try {
          const translatedList = await translateList(block.content)
          results.push({
            original: block.content,
            translated: translatedList,
            type: block.type
          })
        } catch (error) {
          console.error('Failed to translate list:', error)
          results.push({
            original: block.content,
            translated: block.content,
            type: block.type
          })
        }
        break

      case 'paragraph':
      default:
        if (block.content.trim()) {
          try {
            const translated = await translateWithProtection(block.content)
            results.push({
              original: block.content,
              translated,
              type: block.type
            })
          } catch (error) {
            console.error('Failed to translate paragraph:', error)
            results.push({
              original: block.content,
              translated: block.content,
              type: block.type
            })
          }
        }
        break
    }
  }

  // Generate bilingual markdown
  const bilingualMarkdown = generateBilingualMarkdown(results)

  return {
    success: true,
    blocks: results,
    bilingualMarkdown,
    translatedTitle
  }
})

async function translateTable(tableContent: string): Promise<string> {
  const lines = tableContent.split('\n')
  if (lines.length < 2) return tableContent

  const headerLine = lines[0]
  const separatorLine = lines[1]
  const dataLines = lines.slice(2)

  // Translate header
  const headerCells = headerLine.split('|').map(c => c.trim()).filter(Boolean)
  const translatedHeaders = await Promise.all(
    headerCells.map(cell => translateWithProtection(cell))
  )

  // Translate data rows
  const translatedDataLines = await Promise.all(
    dataLines.map(async (line) => {
      const cells = line.split('|').map(c => c.trim()).filter(Boolean)
      const translatedCells = await Promise.all(
        cells.map(cell => translateWithProtection(cell))
      )
      return `| ${translatedCells.join(' | ')} |`
    })
  )

  // Reconstruct table
  const result: string[] = []
  result.push(`| ${translatedHeaders.join(' | ')} |`)
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
        const translated = await translateWithProtection(content)
        return `${indent}${marker} ${translated}`
      }
      return line
    })
  )

  return translatedLines.join('\n')
}

function generateBilingualMarkdown(blocks: BlockResult[]): string {
  const result: string[] = []

  for (const block of blocks) {
    switch (block.type) {
      case 'code':
        result.push(`\`\`\`${block.language || ''}\n${block.original}\n\`\`\``)
        break

      case 'hr':
        result.push('---')
        break

      case 'image':
        result.push(block.original)
        break

      case 'heading':
        result.push(`${'#'.repeat(block.level!)} ${block.original}`)
        result.push('')
        result.push(`${'#'.repeat(block.level!)} ${block.translated}`)
        break

      case 'table':
        // For tables, show translated version inline with original
        result.push(block.original)
        result.push('')
        result.push('**中文翻译:**')
        result.push(block.translated)
        break

      case 'blockquote':
        result.push(':::bilingual')
        result.push(`> ${block.original}`)
        result.push('')
        result.push('---')
        result.push('')
        result.push(`> ${block.translated}`)
        result.push(':::')
        break

      case 'list':
        result.push(':::bilingual')
        result.push(block.original)
        result.push('')
        result.push('---')
        result.push('')
        result.push(block.translated)
        result.push(':::')
        break

      case 'paragraph':
      default:
        if (block.original.trim()) {
          result.push(':::bilingual')
          result.push(block.original)
          result.push('')
          result.push('---')
          result.push('')
          result.push(block.translated)
          result.push(':::')
        }
        break
    }

    result.push('') // Add spacing
  }

  return result.join('\n').trim()
}