// Mock dictionary lookup API
// This can be replaced with a real dictionary API like:
// - Free Dictionary API: https://dictionaryapi.dev/
// - Oxford Dictionaries API
// - Merriam-Webster API

const mockDictionary: Record<string, { phonetic: string; definition: string }> = {
  'sleep': { phonetic: '/sliːp/', definition: 'A condition of body and mind which typically recurs for several hours every night, in which the nervous system is inactive, the eyes closed, the postural muscles relaxed, and consciousness practically suspended.' },
  'essential': { phonetic: '/ɪˈsenʃəl/', definition: 'Absolutely necessary; extremely important.' },
  'health': { phonetic: '/helθ/', definition: 'The state of being free from illness or injury.' },
  'memory': { phonetic: '/ˈmeməri/', definition: 'The faculty by which the mind stores and remembers information.' },
  'learning': { phonetic: '/ˈlɜːrnɪŋ/', definition: 'The acquisition of knowledge or skills through experience, study, or by being taught.' },
  'brain': { phonetic: '/breɪn/', definition: 'An organ of soft nervous tissue contained in the skull of vertebrates, functioning as the coordinating center of sensation and intellectual and nervous activity.' },
  'concentration': { phonetic: '/ˌkɒnsənˈtreɪʃən/', definition: 'The action or power of focusing all one\'s attention.' },
  'vocabulary': { phonetic: '/vəˈkæbjʊləri/', definition: 'The body of words used in a particular language.' },
  'reading': { phonetic: '/ˈriːdɪŋ/', definition: 'The action or skill of reading written or printed matter silently or aloud.' },
  'article': { phonetic: '/ˈɑːrtɪkl/', definition: 'A piece of writing included with others in a newspaper, magazine, or other publication.' },
  'practice': { phonetic: '/ˈpræktɪs/', definition: 'The actual application or use of an idea, belief, or method, as opposed to theories relating to it.' },
  'language': { phonetic: '/ˈlæŋɡwɪdʒ/', definition: 'The method of human communication, either spoken or written, consisting of the use of words in a structured and conventional way.' },
  'english': { phonetic: '/ˈɪŋɡlɪʃ/', definition: 'The West Germanic language of England, now widely used in many varieties throughout the world.' },
  'skill': { phonetic: '/skɪl/', definition: 'The ability to do something well; expertise.' },
  'improve': { phonetic: '/ɪmˈpruːv/', definition: 'Make or become better.' },
  'understand': { phonetic: '/ˌʌndərˈstænd/', definition: 'Perceive the intended meaning of (words, a language, or a speaker).' },
  'word': { phonetic: '/wɜːrd/', definition: 'A single distinct element of speech or writing, used with others (or sometimes alone) to form a sentence.' },
  'sentence': { phonetic: '/ˈsentəns/', definition: 'A set of words that is complete in itself, typically containing a subject and predicate, conveying a statement, question, exclamation, or command.' },
  'study': { phonetic: '/ˈstʌdi/', definition: 'Devote time and attention to acquiring knowledge on an academic subject, especially by means of books.' },
  'knowledge': { phonetic: '/ˈnɒlɪdʒ/', definition: 'Facts, information, and skills acquired through experience or education; the theoretical or practical understanding of a subject.' },
}

export default defineEventHandler(async (event) => {
  const query = getQuery(event)
  const word = (query.word as string || '').toLowerCase().trim()

  if (!word) {
    throw createError({
      statusCode: 400,
      message: 'Word parameter is required'
    })
  }

  // Look up in mock dictionary
  const entry = mockDictionary[word]

  if (entry) {
    return {
      word,
      phonetic: entry.phonetic,
      definition: entry.definition
    }
  }

  // For unknown words, generate a mock response
  // In production, this would call a real dictionary API
  return {
    word,
    phonetic: `/${word}/`,
    definition: `Definition for "${word}" - this is a mock response. In production, this would be fetched from a real dictionary API.`
  }
})