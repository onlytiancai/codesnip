export default defineEventHandler(async (event) => {
  // Get current user from session
  const session = await getUserSession(event)

  if (!session?.user?.id) {
    throw createError({
      statusCode: 401,
      message: 'Unauthorized'
    })
  }

  const userId = session.user.id
  const query = getQuery(event)

  // Pagination
  const page = parseInt(query.page as string) || 1
  const limit = parseInt(query.limit as string) || 20
  const skip = (page - 1) * limit

  // Build filter
  const where: any = { userId }

  if (query.filter === 'mastered') {
    where.progress = 100
  } else if (query.filter === 'learning') {
    where.progress = { lt: 100 }
  }

  // Sorting
  let orderBy: any = { createdAt: 'desc' }
  if (query.sort === 'alpha') {
    orderBy = { word: 'asc' }
  } else if (query.sort === 'progress') {
    orderBy = { progress: 'desc' }
  }

  // Get vocabulary
  const [vocabulary, total] = await Promise.all([
    prisma.vocabulary.findMany({
      where,
      skip,
      take: limit,
      orderBy,
      include: {
        article: {
          select: { id: true, title: true, slug: true }
        }
      }
    }),
    prisma.vocabulary.count({ where })
  ])

  // Get stats
  const [totalWords, mastered, learning] = await Promise.all([
    prisma.vocabulary.count({ where: { userId } }),
    prisma.vocabulary.count({ where: { userId, progress: 100 } }),
    prisma.vocabulary.count({ where: { userId, progress: { lt: 100 } } })
  ])

  return {
    vocabulary: vocabulary.map(v => ({
      id: v.id,
      word: v.word,
      phonetic: v.phonetic,
      definition: v.definition,
      example: v.example,
      progress: v.progress,
      article: v.article,
      createdAt: v.createdAt,
      lastReviewAt: v.lastReviewAt
    })),
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit)
    },
    stats: {
      totalWords,
      mastered,
      learning
    }
  }
})