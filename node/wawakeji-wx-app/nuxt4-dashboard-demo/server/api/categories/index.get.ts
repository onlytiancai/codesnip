export default defineEventHandler(async (event) => {
  const categories = await prisma.category.findMany({
    orderBy: { sortOrder: 'asc' },
    include: {
      _count: {
        select: {
          Article: {
            where: {
              status: 'published'
            }
          }
        }
      }
    }
  })

  return categories.map(cat => ({
    id: cat.id,
    name: cat.name,
    slug: cat.slug,
    icon: cat.icon,
    color: cat.color,
    description: cat.description,
    articleCount: cat._count.Article
  }))
})