// Admin API: List dictionary words with pagination and search
import { z } from 'zod'
import { useDictionaryDb } from '../../../utils/dictionary'

const querySchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
  search: z.string().optional(),
  tag: z.string().optional(), // Filter by tag (zk, gk, cet4, cet6, ielts, toefl, gre, etc.)
  hasPhonetic: z.enum(['true', 'false', 'all']).optional(),
  hasTranslation: z.enum(['true', 'false', 'all']).optional(),
  sortBy: z.enum(['word', 'collins', 'bnc', 'frq']).default('word'),
  sortOrder: z.enum(['asc', 'desc']).default('asc')
})

export default defineEventHandler(async (event) => {
  const result = await getValidatedQuery(event, querySchema.parse)
  const { page, limit, search, tag, hasPhonetic, hasTranslation, sortBy, sortOrder } = result

  const offset = (page - 1) * limit

  // Build where clause
  const where: string[] = []
  if (search) {
    where.push(`word LIKE '%${search.replace(/'/g, "''")}%'`)
  }
  if (tag) {
    // Tags are space-separated, so we search for the tag within the tag field
    where.push(`(tag LIKE '%${tag}%' OR tag LIKE '% ${tag}%' OR tag LIKE '${tag} %')`)
  }
  if (hasPhonetic === 'true') {
    where.push('phonetic IS NOT NULL AND phonetic != ""')
  } else if (hasPhonetic === 'false') {
    where.push('phonetic IS NULL OR phonetic = ""')
  }
  if (hasTranslation === 'true') {
    where.push('translation IS NOT NULL AND translation != ""')
  } else if (hasTranslation === 'false') {
    where.push('translation IS NULL OR translation = ""')
  }

  const whereClause = where.length > 0 ? `WHERE ${where.join(' AND ')}` : ''

  // Get total count
  const countResult = await useDictionaryDb().$queryRawUnsafe<{ count: bigint }[]>(
    `SELECT COUNT(*) as count FROM Dictionary ${whereClause}`
  )
  const total = Number(countResult[0].count)

  // Get paginated results
  const words = await useDictionaryDb().$queryRawUnsafe<any[]>(
    `SELECT * FROM Dictionary ${whereClause} ORDER BY ${sortBy} ${sortOrder.toUpperCase()} LIMIT ${limit} OFFSET ${offset}`
  )

  return {
    words,
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit)
    }
  }
})