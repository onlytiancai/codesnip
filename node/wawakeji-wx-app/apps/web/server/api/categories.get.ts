import { H3Event } from 'h3'
import { prisma } from '~/server/db'

export default defineEventHandler(async (event) => {
  try {
    const categories = await prisma.article.groupBy({
      by: ['category'],
      where: {
        published: true,
      },
      _count: {
        category: true,
      },
    })

    return categories.map((cat) => ({
      id: cat.category,
      name: cat.category,
      count: cat._count.category,
    }))
  } catch (error) {
    console.error('Error fetching categories:', error)
    throw createError({
      statusCode: 500,
      message: 'Failed to fetch categories',
    })
  }
})
