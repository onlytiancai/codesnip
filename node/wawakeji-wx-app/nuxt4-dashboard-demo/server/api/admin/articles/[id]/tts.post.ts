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

  // Return mp3 URLs for testing
  // Frontend will use Web Speech API for actual playback if audio URL is not playable
  const ttsParagraphs = (paragraphs || []).map((p: { order: number; en: string }) => ({
    order: p.order,
    en: p.en,
    audio: '/mp3/nice_to_meet_you.mp3'
  }))

  const ttsSentences = (sentences || []).map((s: { order: number; paragraphIndex?: number; en: string }) => ({
    order: s.order,
    paragraphIndex: s.paragraphIndex,
    en: s.en,
    audio: '/mp3/nice_to_meet_you.mp3'
  }))

  return {
    paragraphs: ttsParagraphs,
    sentences: ttsSentences
  }
})