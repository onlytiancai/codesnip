export default defineEventHandler(async (event) => {
  const body = await readBody(event)

  const { paragraphs, sentences } = body

  // Fake TTS - return placeholder URLs
  // Frontend will use Web Speech API for actual playback
  const ttsParagraphs = (paragraphs || []).map((p: { order: number; en: string }) => ({
    order: p.order,
    en: p.en,
    audio: `tts://placeholder/new/paragraph/${p.order}`
  }))

  const ttsSentences = (sentences || []).map((s: { order: number; paragraphIndex?: number; en: string }) => ({
    order: s.order,
    paragraphIndex: s.paragraphIndex,
    en: s.en,
    audio: `tts://placeholder/new/sentence/${s.order}`
  }))

  return {
    paragraphs: ttsParagraphs,
    sentences: ttsSentences
  }
})