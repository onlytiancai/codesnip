
export default defineEventHandler(async (event) => {
  const query = getQuery(event)

  // Build filter conditions
  const where: any = {}

  if (query.search) {
    where.name = {
      contains: query.search as string
    }
  }

  // Get tags with article count
  const tags = await prisma.tag.findMany({
    where,
    orderBy: { name: 'asc' },
    include: {
      _count: {
        select: { articles: true }
      }
    }
  })

  return tags.map(tag => ({
    ...tag,
    articleCount: tag._count.articles
  }))
})