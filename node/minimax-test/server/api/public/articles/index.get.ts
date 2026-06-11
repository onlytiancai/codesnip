import { prisma } from '../../../utils/db'

export default defineEventHandler(async (event) => {
  const query = getQuery(event)
  const categorySlug = query.category as string | undefined
  const tagSlug = query.tag as string | undefined
  const limit = query.limit ? parseInt(query.limit as string) : 20
  const offset = query.offset ? parseInt(query.offset as string) : 0

  const where: Record<string, unknown> = {
    isPublished: true
  }

  if (categorySlug) {
    where.category = {
      slug: categorySlug
    }
  }

  if (tagSlug) {
    where.tags = {
      some: {
        slug: tagSlug
      }
    }
  }

  const [articles, total] = await Promise.all([
    prisma.article.findMany({
      where,
      include: {
        category: true,
        tags: true,
        user: {
          select: {
            id: true,
            name: true
          }
        }
      },
      orderBy: {
        createdAt: 'desc'
      },
      take: limit,
      skip: offset
    }),
    prisma.article.count({ where })
  ])

  return {
    articles,
    total,
    limit,
    offset
  }
})
