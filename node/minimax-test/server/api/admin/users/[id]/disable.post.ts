import { prisma } from '../../../../utils/db'

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user || session.user.id !== '1') {
    throw createError({ statusCode: 403, message: 'Admin only' })
  }

  const userId = getRouterParam(event, 'id')

  if (userId === '1') {
    throw createError({ statusCode: 400, message: 'Cannot disable admin user' })
  }

  const user = await prisma.user.update({
    where: { id: userId },
    data: { isDisabled: true }
  })

  return {
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      isDisabled: user.isDisabled
    }
  }
})
