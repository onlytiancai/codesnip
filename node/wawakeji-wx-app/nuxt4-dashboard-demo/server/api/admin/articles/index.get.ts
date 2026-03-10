
export default defineEventHandler(async (event) => {
  const query = getQuery(event)

  // Pagination
  const page = parseInt(query.page as string) || 1
  const limit = parseInt(query.limit as string) || 10
  const skip = (page - 1) * limit

  // Build filter conditions
  const where: any = {}

  if (query.status && query.status !== 'all') {
    where.status = query.status
  }

  if (query.categoryId && query.categoryId !== 'all') {
    where.categoryId = parseInt(query.categoryId as string)
  }

  if (query.difficulty && query.difficulty !== 'all') {
    where.difficulty = query.difficulty
  }

  if (query.search) {
    where.OR = [
      { title: { contains: query.search as string } },
      { excerpt: { contains: query.search as string } }
    ]
  }

  // Get articles with pagination
  const [articles, total] = await Promise.all([
    prisma.article.findMany({
      where,
      skip,
      take: limit,
      orderBy: { createdAt: 'desc' },
      include: {
        Category: true,
        ArticleTag: {
          include: {
            Tag: true
          }
        },
        User: {
          select: {
            id: true,
            name: true,
            email: true
          }
        },
        _count: {
          select: { Sentence: true }
        }
      }
    }),
    prisma.article.count({ where })
  ])

  return {
    articles: articles.map(article => ({
      ...article,
      category: article.Category,
      tags: article.ArticleTag.map(t => t.Tag),
      author: article.User,
      sentenceCount: article._count.Sentence
    })),
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit)
    }
  }
})