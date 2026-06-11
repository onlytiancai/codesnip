import { prisma } from '../../utils/db'

export default defineEventHandler(async (event) => {
  const tags = await prisma.tag.findMany({
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
    tags: tags.map(t => ({
      ...t,
      articleCount: t._count.articles
    }))
  }
})
