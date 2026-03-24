import { prisma } from '../../../../utils/db'

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user || session.user.id !== '1') {
    throw createError({ statusCode: 403, message: 'Admin only' })
  }

  const userId = getRouterParam(event, 'id')

  const user = await prisma.user.findUnique({
    where: { id: userId },
    include: {
      articles: {
        orderBy: { createdAt: 'desc' }
      }
    }
  })

  if (!user) {
    throw createError({ statusCode: 404, message: 'User not found' })
  }

  return {
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      isDisabled: user.isDisabled
    },
    articles: user.articles
  }
})
