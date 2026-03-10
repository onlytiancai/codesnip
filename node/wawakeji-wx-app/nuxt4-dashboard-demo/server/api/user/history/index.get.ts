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
  const limit = parseInt(query.limit as string) || 10
  const skip = (page - 1) * limit

  // Get reading history with article details
  const [history, total] = await Promise.all([
    prisma.readingHistory.findMany({
      where: { userId },
      skip,
      take: limit,
      orderBy: { lastReadAt: 'desc' },
      include: {
        article: {
          include: {
            category: {
              select: { id: true, name: true, slug: true }
            }
          }
        }
      }
    }),
    prisma.readingHistory.count({ where: { userId } })
  ])

  return {
    history: history.map(h => ({
      id: h.id,
      articleId: h.articleId,
      title: h.article.title,
      slug: h.article.slug,
      cover: h.article.cover,
      excerpt: h.article.excerpt,
      difficulty: h.article.difficulty,
      category: h.article.category,
      progress: h.progress,
      lastReadAt: h.lastReadAt,
      completedAt: h.completedAt,
      readTime: Math.ceil(h.article.content?.split(' ').length / 200 || 8) // Estimated read time
    })),
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit)
    }
  }
})