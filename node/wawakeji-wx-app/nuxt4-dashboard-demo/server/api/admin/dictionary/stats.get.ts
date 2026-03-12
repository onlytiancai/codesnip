// Admin API: Get dictionary statistics
import { useDictionaryDb } from '../../../utils/dictionary'

export default defineEventHandler(async () => {
  const db = useDictionaryDb()

  // Get total count
  const totalResult = await db.$queryRawUnsafe<{ count: bigint }[]>(
    'SELECT COUNT(*) as count FROM Dictionary'
  )
  const total = Number(totalResult[0].count)

  // Get words with phonetic
  const phoneticResult = await db.$queryRawUnsafe<{ count: bigint }[]>(
    "SELECT COUNT(*) as count FROM Dictionary WHERE phonetic IS NOT NULL AND phonetic != ''"
  )
  const withPhonetic = Number(phoneticResult[0].count)

  // Get words with translation
  const translationResult = await db.$queryRawUnsafe<{ count: bigint }[]>(
    "SELECT COUNT(*) as count FROM Dictionary WHERE translation IS NOT NULL AND translation != ''"
  )
  const withTranslation = Number(translationResult[0].count)

  // Get words with definition
  const definitionResult = await db.$queryRawUnsafe<{ count: bigint }[]>(
    "SELECT COUNT(*) as count FROM Dictionary WHERE definition IS NOT NULL AND definition != ''"
  )
  const withDefinition = Number(definitionResult[0].count)

  // Get Collins 5-star words
  const collinsResult = await db.$queryRawUnsafe<{ count: bigint }[]>(
    'SELECT COUNT(*) as count FROM Dictionary WHERE collins = 5'
  )
  const collins5Star = Number(collinsResult[0].count)

  // Get Oxford 3000 words
  const oxfordResult = await db.$queryRawUnsafe<{ count: bigint }[]>(
    'SELECT COUNT(*) as count FROM Dictionary WHERE oxford = 1'
  )
  const oxford3000 = Number(oxfordResult[0].count)

  // Get tag statistics
  const tagStats = await db.$queryRawUnsafe<{ tag: string; count: bigint }[]>(
    `SELECT tag, COUNT(*) as count FROM Dictionary
     WHERE tag IS NOT NULL AND tag != ''
     GROUP BY tag
     ORDER BY count DESC
     LIMIT 20`
  )

  // Parse tag stats (tags are space-separated)
  const tagCounts: Record<string, number> = {}
  for (const row of tagStats) {
    if (row.tag) {
      const tags = row.tag.split(/\s+/).filter(t => t)
      for (const t of tags) {
        tagCounts[t] = (tagCounts[t] || 0) + Number(row.count)
      }
    }
  }

  // Tag labels
  const tagLabels: Record<string, string> = {
    'zk': '中考',
    'gk': '高考',
    'cet4': '四级',
    'cet6': '六级',
    'ielts': '雅思',
    'toefl': '托福',
    'gre': 'GRE',
    'ky': '考研',
    'bec': 'BEC',
    'tem4': '专四',
    'tem8': '专八'
  }

  const formattedTagStats = Object.entries(tagCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([tag, count]) => ({
      tag,
      label: tagLabels[tag] || tag.toUpperCase(),
      count
    }))

  return {
    total,
    withPhonetic,
    withTranslation,
    withDefinition,
    collins5Star,
    oxford3000,
    tagStats: formattedTagStats
  }
})