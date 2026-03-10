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

  // Get bookmarks with article details
  const [bookmarks, total] = await Promise.all([
    prisma.bookmark.findMany({
      where: { userId },
      skip,
      take: limit,
      orderBy: { createdAt: 'desc' },
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
    prisma.bookmark.count({ where: { userId } })
  ])

  return {
    bookmarks: bookmarks.map(b => ({
      id: b.id,
      articleId: b.articleId,
      title: b.article.title,
      slug: b.article.slug,
      cover: b.article.cover,
      excerpt: b.article.excerpt,
      difficulty: b.article.difficulty,
      category: b.article.category,
      createdAt: b.createdAt,
      readTime: Math.ceil(b.article.content?.split(' ').length / 200 || 8)
    })),
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit)
    }
  }
})