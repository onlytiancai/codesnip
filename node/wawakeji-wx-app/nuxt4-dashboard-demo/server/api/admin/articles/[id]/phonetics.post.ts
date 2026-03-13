import { phonemize } from 'phonemize'

// Convert American IPA to British IPA
function americanToBritishIPA(ipa: string): string {
  const rules: [RegExp, string][] = [
    // r-color vowels
    [/ɝ/g, 'ɜː'],       // stressed
    [/ɚ/g, 'ə'],        // unstressed
    [/ər/g, 'ə'],       // non-stressed er
    [/ɝː/g, 'ɜː'],      // stressed r-color

    // dark L -> light L
    [/ɫ/g, 'l'],

    // diphthongs
    [/oʊ/g, 'əʊ'],      // GOAT
    [/aɪ/g, 'aɪ'],      // unchanged
    [/eɪ/g, 'eɪ'],      // FACE
    [/ɔɪ/g, 'ɔɪ'],      // CHOICE
    [/aʊ/g, 'aʊ'],      // MOUTH

    // monophthongs
    [/æ/g, 'æ'],        // TRAP
    [/ɑ/g, 'ɒ'],        // LOT
    [/ɪ/g, 'ɪ'],        // KIT
    [/ʌ/g, 'ʌ'],        // STRUT
    [/ʊ/g, 'ʊ'],        // FOOT
    [/ɝ/g, 'ɜː'],       // extra handling

    // silent r (non-stressed word ending)
    [/r\b/g, ''],

    // special consonant combinations
    [/tʃr/g, 'tr'],
    [/dʒr/g, 'dr'],

    // American t flapping -> British t
    [/ɾ/g, 't'],

    // American /ɝ/ followed by l -> British /ɜːl/
    [/ɝl/g, 'ɜːl']
  ]

  let result = ipa
  for (const [pattern, replacement] of rules) {
    result = result.replace(pattern, replacement)
  }
  return result
}

export default defineEventHandler(async (event) => {
  const articleId = parseInt(getRouterParam(event, 'id') || '0')
  const body = await readBody(event)
  const db = useDictionaryDb()

  if (!articleId) {
    throw createError({
      statusCode: 400,
      message: 'Invalid article ID'
    })
  }

  const { items } = body

  if (!items || !Array.isArray(items)) {
    throw createError({
      statusCode: 400,
      message: 'Items array is required'
    })
  }

  // Process each item (sentence) and generate phonetics in word order
  const results = await Promise.all(
    items.map(async (item: { order: number; en: string }) => {
      // Tokenize sentence preserving original order (words + punctuation/numbers)
      const tokens: { text: string; isWord: boolean }[] = []
      const regex = /(\w+)|([^\w\s]+)/g
      let match

      while ((match = regex.exec(item.en)) !== null) {
        tokens.push({
          text: match[0],
          isWord: !!match[1]
        })
      }

      // Get unique words for batch query
      const uniqueWords = [...new Set(
        tokens
          .filter(t => t.isWord)
          .map(t => t.text.toLowerCase())
      )]

      // Batch query phonetics from database
      const phoneticsMap = new Map<string, string>()

      if (uniqueWords.length > 0) {
        try {
          const entries = await db.dictionary.findMany({
            where: {
              word: { in: uniqueWords }
            },
            select: {
              word: true,
              phonetic: true
            }
          })

          for (const entry of entries) {
            // Only use dictionary phonetic if it's not empty
            if (entry.phonetic && entry.phonetic.trim()) {
              phoneticsMap.set(entry.word.toLowerCase(), entry.phonetic.trim())
            }
          }
        } catch (e) {
          // Ignore lookup errors
        }
      }

      // Generate phonetics for words not in dictionary using phonemize
      for (const word of uniqueWords) {
        if (!phoneticsMap.has(word)) {
          try {
            const americanIPA = phonemize(word, { stripStress: true })
            if (americanIPA && americanIPA.trim()) {
              const britishIPA = americanToBritishIPA(americanIPA.trim())
              phoneticsMap.set(word, britishIPA)
            }
          } catch (e) {
            // Ignore phonemize errors
          }
        }
      }

      // Build phonetics array in original sentence order
      const phonetics = tokens.map(token => {
        if (token.isWord) {
          const lowerWord = token.text.toLowerCase()
          const phonetic = phoneticsMap.get(lowerWord)
          return {
            text: token.text,
            word: lowerWord,
            phonetic: phonetic ?? '' // Empty string for words that couldn't be processed
          }
        } else {
          // Non-word characters (punctuation, numbers)
          return {
            text: token.text,
            word: token.text.toLowerCase(),
            phonetic: null as string | null // null for non-words
          }
        }
      })

      return {
        order: item.order,
        phonetics
      }
    })
  )

  return { items: results }
})