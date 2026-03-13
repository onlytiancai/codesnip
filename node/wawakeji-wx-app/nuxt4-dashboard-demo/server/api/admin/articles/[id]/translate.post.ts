export default defineEventHandler(async (event) => {
  const articleId = parseInt(getRouterParam(event, 'id') || '0')
  const body = await readBody(event)

  if (!articleId) {
    throw createError({
      statusCode: 400,
      message: 'Invalid article ID'
    })
  }

  const { paragraphs, sentences } = body

  // Fake translation - return placeholder translations
  const translatedParagraphs = (paragraphs || []).map((p: { order: number; en: string }) => ({
    order: p.order,
    en: p.en,
    cn: `[翻译: ${p.en.slice(0, 50)}${p.en.length > 50 ? '...' : ''}]`
  }))

  const translatedSentences = (sentences || []).map((s: { order: number; paragraphIndex?: number; en: string }) => ({
    order: s.order,
    paragraphIndex: s.paragraphIndex,
    en: s.en,
    cn: `[翻译: ${s.en}]`
  }))

  return {
    paragraphs: translatedParagraphs,
    sentences: translatedSentences
  }
})