import { H3Event } from 'h3'
import { prisma } from '~/server/db'

export default defineEventHandler(async (event) => {
  try {
    const query = getQuery(event)
    const page = Number(query.page) || 1
    const pageSize = Number(query.pageSize) || 10
    const category = query.category as string | undefined
    const difficulty = query.difficulty as string | undefined
    const search = query.search as string | undefined

    const skip = (page - 1) * pageSize

    const where: any = {
      published: true,
    }

    if (category) {
      where.category = category
    }

    if (difficulty) {
      where.difficulty = difficulty
    }

    if (search) {
      where.OR = [
        { title: { contains: search } },
        { summary: { contains: search } },
      ]
    }

    const [articles, total] = await Promise.all([
      prisma.article.findMany({
        where,
        skip,
        take: pageSize,
        orderBy: { publishedAt: 'desc' },
        select: {
          id: true,
          title: true,
          slug: true,
          summary: true,
          category: true,
          coverImage: true,
          difficulty: true,
          publishedAt: true,
          createdAt: true,
        },
      }),
      prisma.article.count({ where }),
    ])

    return {
      items: articles,
      total,
      page,
      pageSize,
      hasMore: skip + articles.length < total,
    }
  } catch (error) {
    console.error('Error fetching articles:', error)
    throw createError({
      statusCode: 500,
      message: 'Failed to fetch articles',
    })
  }
})
