
export default defineEventHandler(async (event) => {
  const query = getQuery(event)
  const days = parseInt(query.days as string) || 30

  // Calculate date range
  const startDate = new Date()
  startDate.setDate(startDate.getDate() - days)

  // Get user registrations by day
  const userRegistrations = await prisma.$queryRaw<Array<{ date: string; count: bigint }>>`
    SELECT date(createdAt) as date, COUNT(*) as count
    FROM User
    WHERE createdAt >= datetime(${startDate.toISOString()})
    GROUP BY date(createdAt)
    ORDER BY date(createdAt) ASC
  `

  // Get article creations by day
  const articleCreations = await prisma.$queryRaw<Array<{ date: string; count: bigint }>>`
    SELECT date(createdAt) as date, COUNT(*) as count
    FROM Article
    WHERE createdAt >= datetime(${startDate.toISOString()})
    GROUP BY date(createdAt)
    ORDER BY date(createdAt) ASC
  `

  // Get category distribution
  const categoryDistribution = await prisma.category.findMany({
    include: {
      _count: {
        select: { articles: true }
      }
    },
    orderBy: {
      articles: { _count: 'desc' }
    },
    take: 10
  })

  // Get difficulty distribution
  const difficultyDistribution = await prisma.article.groupBy({
    by: ['difficulty'],
    _count: { id: true }
  })

  // Get top articles by views
  const topArticles = await prisma.article.findMany({
    take: 5,
    orderBy: { views: 'desc' },
    include: {
      category: true
    }
  })

  // Get tag usage
  const tagUsage = await prisma.tag.findMany({
    include: {
      _count: {
        select: { articles: true }
      }
    },
    orderBy: {
      articles: { _count: 'desc' }
    },
    take: 10
  })

  return {
    userRegistrations: userRegistrations.map(r => ({
      date: r.date,
      count: Number(r.count)
    })),
    articleCreations: articleCreations.map(r => ({
      date: r.date,
      count: Number(r.count)
    })),
    categoryDistribution: categoryDistribution.map(c => ({
      name: c.name,
      count: c._count.articles
    })),
    difficultyDistribution: difficultyDistribution.map(d => ({
      difficulty: d.difficulty,
      count: d._count.id
    })),
    topArticles,
    tagUsage: tagUsage.map(t => ({
      ...t,
      articleCount: t._count.articles
    }))
  }
})