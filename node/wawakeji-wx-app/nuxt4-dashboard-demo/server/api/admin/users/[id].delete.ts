
export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')
  const session = await getUserSession(event)

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid user ID'
    })
  }

  // Prevent deleting yourself
  if (session?.user?.id === id) {
    throw createError({
      statusCode: 400,
      message: 'You cannot delete your own account'
    })
  }

  // Check if user exists
  const existing = await prisma.user.findUnique({
    where: { id }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: 'User not found'
    })
  }

  // Delete user (accounts and articles will be cascade deleted)
  await prisma.user.delete({
    where: { id }
  })

  return { success: true }
})