import { prisma } from '../../../../utils/db'

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user || session.user.id !== '1') {
    throw createError({ statusCode: 403, message: 'Admin only' })
  }

  const articleId = getRouterParam(event, 'id')

  const article = await prisma.article.update({
    where: { id: articleId },
    data: { isPublished: false }
  })

  return { article }
})
