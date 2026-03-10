export default defineEventHandler(async (event) => {
  const query = getQuery(event)

  // Pagination
  const page = parseInt(query.page as string) || 1
  const limit = parseInt(query.limit as string) || 9
  const skip = (page - 1) * limit

  // Build filter conditions - only published articles
  const where: any = {
    status: 'published'
  }

  if (query.categorySlug) {
    where.Category = {
      slug: query.categorySlug
    }
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
            name: true
          }
        }
      }
    }),
    prisma.article.count({ where })
  ])

  return {
    articles: articles.map(article => ({
      id: article.id,
      title: article.title,
      slug: article.slug,
      excerpt: article.excerpt,
      cover: article.cover,
      difficulty: article.difficulty,
      category: article.Category,
      tags: article.ArticleTag.map(t => t.Tag),
      author: article.User,
      views: article.views,
      bookmarks: article.bookmarks,
      readTime: Math.ceil(article.content.split(' ').length / 200), // Estimate read time
      createdAt: article.createdAt
    })),
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit)
    }
  }
})