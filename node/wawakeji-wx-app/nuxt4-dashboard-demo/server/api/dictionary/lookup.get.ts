// Dictionary lookup API using ECDICT database (separate database)
// Returns phonetic, English definition, Chinese translation, and audio URL

// Try to find the lemma (base form) of a word
async function findLemma(word: string): Promise<string | null> {
  const db = useDictionaryDb()

  // Common word form patterns
  const patterns = [
    // -ed past tense/participle
    { suffix: 'ed', replace: '' },
    { suffix: 'ied', replace: 'y' },
    // -ing present participle
    { suffix: 'ing', replace: '' },
    { suffix: 'ing', replace: 'e' },
    // -s third person
    { suffix: 's', replace: '' },
    { suffix: 'es', replace: '' },
    { suffix: 'ies', replace: 'y' },
    // -er/-est comparative/superlative
    { suffix: 'er', replace: '' },
    { suffix: 'er', replace: 'e' },
    { suffix: 'est', replace: '' },
    { suffix: 'est', replace: 'e' },
    // -ly adverb
    { suffix: 'ly', replace: '' },
    { suffix: 'ily', replace: 'y' },
  ]

  for (const pattern of patterns) {
    if (word.endsWith(pattern.suffix)) {
      const possibleBase = word.slice(0, -pattern.suffix.length) + pattern.replace
      if (possibleBase.length >= 2) {
        const exists = await db.dictionary.findUnique({
          where: { word: possibleBase }
        })
        if (exists) {
          return possibleBase
        }
      }
    }
  }

  return null
}

export default defineEventHandler(async (event) => {
  const query = getQuery(event)
  const word = (query.word as string || '').toLowerCase().trim()
  const db = useDictionaryDb()

  if (!word) {
    throw createError({
      statusCode: 400,
      message: 'Word parameter is required'
    })
  }

  // Try to find the word directly
  let entry = await db.dictionary.findUnique({
    where: { word }
  })

  // If not found, try to find the lemma
  let lemma: string | null = null
  if (!entry) {
    lemma = await findLemma(word)
    if (lemma) {
      entry = await db.dictionary.findUnique({
        where: { word: lemma }
      })
    }
  }

  if (entry) {
    // Youdao Dictionary audio URLs
    const audioUs = `https://dict.youdao.com/dictvoice?type=0&audio=${encodeURIComponent(entry.word)}`
    const audioUk = `https://dict.youdao.com/dictvoice?type=1&audio=${encodeURIComponent(entry.word)}`

    return {
      word: entry.word,
      phonetic: entry.phonetic || '',
      definition: entry.definition || '',
      translation: entry.translation || '',
      pos: entry.pos || '',
      audioUs,
      audioUk,
      lemma: lemma || undefined,
      exchange: entry.exchange || undefined
    }
  }

  // Word not found in database - return basic info with audio URLs
  const audioUs = `https://dict.youdao.com/dictvoice?type=0&audio=${encodeURIComponent(word)}`
  const audioUk = `https://dict.youdao.com/dictvoice?type=1&audio=${encodeURIComponent(word)}`

  return {
    word,
    phonetic: '',
    definition: '',
    translation: '',
    pos: '',
    audioUs,
    audioUk,
    found: false
  }
})