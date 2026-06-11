import { prisma } from '../../../utils/db'

export default defineEventHandler(async (event) => {
  const id = getRouterParam(event, 'id')

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Article ID is required'
    })
  }

  const article = await prisma.article.findUnique({
    where: { id },
    include: {
      category: true,
      tags: true,
      images: true,
      user: {
        select: {
          id: true,
          name: true
        }
      }
    }
  })

  if (!article || !article.isPublished) {
    throw createError({
      statusCode: 404,
      message: 'Article not found'
    })
  }

  return { article }
})
