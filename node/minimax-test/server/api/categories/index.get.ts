import { prisma } from '../../utils/db'

export default defineEventHandler(async (event) => {
  const categories = await prisma.category.findMany({
    include: {
      _count: {
        select: {
          articles: true
        }
      }
    },
    orderBy: {
      name: 'asc'
    }
  })

  return {
    categories: categories.map(c => ({
      ...c,
      articleCount: c._count.articles
    }))
  }
})
