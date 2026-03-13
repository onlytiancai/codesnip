export default defineEventHandler(async (event) => {
  const body = await readBody(event)

  const { paragraphs, sentences } = body

  // Fake translation - return placeholder translations
  const translatedParagraphs = (paragraphs || []).map((p: { order: number; en: string }) => ({
    order: p.order,
    en: p.en,
    cn: `[Translation: ${p.en.slice(0, 50)}${p.en.length > 50 ? '...' : ''}]`
  }))

  const translatedSentences = (sentences || []).map((s: { order: number; paragraphIndex?: number; en: string }) => ({
    order: s.order,
    paragraphIndex: s.paragraphIndex,
    en: s.en,
    cn: `[Translation: ${s.en}]`
  }))

  return {
    paragraphs: translatedParagraphs,
    sentences: translatedSentences
  }
})