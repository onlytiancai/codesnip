import { prisma } from '../../utils/db'

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user || session.user.id !== '1') {
    throw createError({ statusCode: 403, message: 'Admin only' })
  }

  const users = await prisma.user.findMany({
    include: {
      _count: {
        select: { articles: true }
      }
    },
    orderBy: { createdAt: 'desc' }
  })

  return {
    users: users.map(u => ({
      id: u.id,
      email: u.email,
      name: u.name,
      isDisabled: u.isDisabled,
      createdAt: u.createdAt,
      articleCount: u._count.articles
    }))
  }
})
