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
        Article: {
          include: {
            Category: {
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
      title: b.Article.title,
      slug: b.Article.slug,
      cover: b.Article.cover,
      excerpt: b.Article.excerpt,
      difficulty: b.Article.difficulty,
      category: b.Article.Category,
      createdAt: b.createdAt,
      readTime: Math.ceil(b.Article.content?.split(' ').length / 200 || 8)
    })),
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit)
    }
  }
})