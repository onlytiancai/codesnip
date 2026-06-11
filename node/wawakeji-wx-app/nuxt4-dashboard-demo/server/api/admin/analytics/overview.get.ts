
export default defineEventHandler(async (event) => {
  // Get total counts
  const [
    totalUsers,
    totalArticles,
    publishedArticles,
    totalCategories,
    totalTags,
    totalViews,
    totalBookmarks
  ] = await Promise.all([
    prisma.user.count(),
    prisma.article.count(),
    prisma.article.count({ where: { status: 'published' } }),
    prisma.category.count({ where: { status: 'active' } }),
    prisma.tag.count(),
    prisma.article.aggregate({
      _sum: { views: true }
    }),
    prisma.article.aggregate({
      _sum: { bookmarks: true }
    })
  ])

  // Get new users in last 30 days
  const thirtyDaysAgo = new Date()
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30)

  const newUsersLast30Days = await prisma.user.count({
    where: {
      createdAt: {
        gte: thirtyDaysAgo
      }
    }
  })

  const newArticlesLast30Days = await prisma.article.count({
    where: {
      createdAt: {
        gte: thirtyDaysAgo
      }
    }
  })

  // Get recent articles
  const recentArticles = await prisma.article.findMany({
    take: 5,
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
  })

  // Get recent users
  const recentUsers = await prisma.user.findMany({
    take: 5,
    orderBy: { createdAt: 'desc' },
    select: {
      id: true,
      name: true,
      email: true,
      avatar: true,
      createdAt: true
    }
  })

  return {
    stats: {
      totalUsers,
      totalArticles,
      publishedArticles,
      totalCategories,
      totalTags,
      totalViews: totalViews._sum.views || 0,
      totalBookmarks: totalBookmarks._sum.bookmarks || 0,
      newUsersLast30Days,
      newArticlesLast30Days
    },
    recentArticles,
    recentUsers
  }
})