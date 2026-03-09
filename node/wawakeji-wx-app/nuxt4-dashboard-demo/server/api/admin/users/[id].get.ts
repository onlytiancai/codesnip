
export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid user ID'
    })
  }

  const user = await prisma.user.findUnique({
    where: { id },
    select: {
      id: true,
      email: true,
      name: true,
      avatar: true,
      role: true,
      createdAt: true,
      updatedAt: true,
      accounts: {
        select: {
          provider: true
        }
      },
      _count: {
        select: { articles: true }
      }
    }
  })

  if (!user) {
    throw createError({
      statusCode: 404,
      message: 'User not found'
    })
  }

  return {
    ...user,
    articleCount: user._count.articles,
    oauthProviders: user.accounts.map(a => a.provider)
  }
})