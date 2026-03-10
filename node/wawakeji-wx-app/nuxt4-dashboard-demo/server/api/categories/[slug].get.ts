export default defineEventHandler(async (event) => {
  const slug = getRouterParam(event, 'slug')

  if (!slug) {
    throw createError({
      statusCode: 400,
      message: 'Category slug is required'
    })
  }

  const query = getQuery(event)

  // Pagination
  const page = parseInt(query.page as string) || 1
  const limit = parseInt(query.limit as string) || 9
  const skip = (page - 1) * limit

  const category = await prisma.category.findUnique({
    where: { slug }
  })

  if (!category) {
    throw createError({
      statusCode: 404,
      message: 'Category not found'
    })
  }

  // Build filter conditions
  const where: any = {
    status: 'published',
    categoryId: category.id
  }

  if (query.difficulty && query.difficulty !== 'all') {
    where.difficulty = query.difficulty
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
    category: {
      id: category.id,
      name: category.name,
      slug: category.slug,
      icon: category.icon,
      color: category.color,
      description: category.description
    },
    articles: articles.map(article => ({
      id: article.id,
      title: article.title,
      slug: article.slug,
      excerpt: article.excerpt,
      cover: article.cover,
      difficulty: article.difficulty,
      views: article.views,
      bookmarks: article.bookmarks,
      readTime: Math.ceil(article.content.split(' ').length / 200),
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