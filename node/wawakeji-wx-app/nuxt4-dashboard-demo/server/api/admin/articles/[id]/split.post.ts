export default defineEventHandler(async (event) => {
  const articleId = parseInt(getRouterParam(event, 'id') || '0')
  const body = await readBody(event)

  if (!articleId) {
    throw createError({
      statusCode: 400,
      message: 'Invalid article ID'
    })
  }

  if (!body.content || typeof body.content !== 'string') {
    throw createError({
      statusCode: 400,
      message: 'Content is required'
    })
  }

  const content = body.content

  // Split content into paragraphs (by double newlines)
  const paragraphTexts = content
    .split(/\n\n+/)
    .map(p => p.trim())
    .filter(p => p.length > 0)

  // Build paragraphs and sentences arrays
  const paragraphs: Array<{ order: number; en: string }> = []
  const sentences: Array<{ order: number; paragraphIndex: number; en: string }> = []

  let globalSentenceOrder = 0

  paragraphTexts.forEach((paragraphText, pIndex) => {
    // Add paragraph
    paragraphs.push({
      order: pIndex,
      en: paragraphText
    })

    // Split paragraph into sentences
    const sentenceTexts = paragraphText
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim())
      .filter(s => s.length > 0)

    sentenceTexts.forEach((sentenceText) => {
      sentences.push({
        order: globalSentenceOrder++,
        paragraphIndex: pIndex,
        en: sentenceText
      })
    })
  })

  return {
    paragraphs,
    sentences
  }
})