
export default defineEventHandler(async (event) => {
  const query = getQuery(event)

  // Build filter conditions
  const where: any = {}

  if (query.status && query.status !== 'all') {
    where.status = query.status
  }

  // Get categories with article count
  const categories = await prisma.category.findMany({
    where,
    orderBy: { sortOrder: 'asc' },
    include: {
      _count: {
        select: { articles: true }
      }
    }
  })

  return categories.map(cat => ({
    ...cat,
    articleCount: cat._count.articles
  }))
})