// Phonetics API using ECDICT database (separate database)
// Returns phonetic transcriptions for each word in a text

export default defineEventHandler(async (event) => {
  const query = getQuery(event)
  const text = (query.text as string || '').trim()
  const db = useDictionaryDb()

  if (!text) {
    throw createError({
      statusCode: 400,
      message: 'Text parameter is required'
    })
  }

  // Split text into words
  const wordRegex = /(\w+)|([^\w\s]+)/g
  const matches = [...text.matchAll(wordRegex)]

  // Get unique words for batch query
  const uniqueWords = new Set<string>()
  for (const match of matches) {
    const word = match[1]?.toLowerCase()
    if (word && word.length >= 1) {
      uniqueWords.add(word)
    }
  }

  // Batch query phonetics from database
  const phoneticsMap = new Map<string, string>()
  if (uniqueWords.size > 0) {
    const entries = await db.dictionary.findMany({
      where: {
        word: { in: Array.from(uniqueWords) }
      },
      select: {
        word: true,
        phonetic: true
      }
    })

    for (const entry of entries) {
      if (entry.phonetic) {
        phoneticsMap.set(entry.word, entry.phonetic)
      }
    }
  }

  // Build result with phonetics
  const words = matches.map(match => {
    const text = match[0]
    const isWord = !!match[1]
    const clean = isWord ? text.toLowerCase() : ''

    return {
      word: clean,
      original: text,
      phonetic: isWord ? (phoneticsMap.get(clean) || '') : ''
    }
  })

  return { words }
})