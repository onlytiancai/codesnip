import { prisma } from '../../utils/db'

export default defineEventHandler(async (event) => {
  const id = getRouterParam(event, 'id')
  const session = await getUserSession(event)

  if (!session?.user) {
    throw createError({
      statusCode: 401,
      message: 'Not authenticated'
    })
  }

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Article ID is required'
    })
  }

  const article = await prisma.article.findUnique({
    where: { id }
  })

  if (!article) {
    throw createError({
      statusCode: 404,
      message: 'Article not found'
    })
  }

  if (article.userId !== session.user.id) {
    throw createError({
      statusCode: 403,
      message: 'Access denied'
    })
  }

  const body = await readBody(event)

  const updatedArticle = await prisma.article.update({
    where: { id },
    data: {
      title: body.title ?? article.title,
      content: body.content ?? article.content,
      description: body.description ?? article.description
    },
    include: {
      category: true,
      tags: true,
      images: true
    }
  })

  return { article: updatedArticle }
})
