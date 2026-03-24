import { prisma } from '../../utils/db'

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
          name: true,
          email: true
        }
      }
    }
  })

  if (!article) {
    throw createError({
      statusCode: 404,
      message: 'Article not found'
    })
  }

  const session = await getUserSession(event)

  if (!article.isPublished && (!session?.user || session.user.id !== article.userId)) {
    throw createError({
      statusCode: 403,
      message: 'Access denied'
    })
  }

  return { article }
})
