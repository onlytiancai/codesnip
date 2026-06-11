export default defineEventHandler(async (event) => {
  // Get current user from session
  const session = await getUserSession(event)

  if (!session?.user?.id) {
    throw createError({
      statusCode: 401,
      message: 'Unauthorized'
    })
  }

  const userId = session.user.id
  const vocabId = parseInt(event.context.params?.id || '0')

  if (!vocabId) {
    throw createError({
      statusCode: 400,
      message: 'Vocabulary ID is required'
    })
  }

  // Delete vocabulary (only if belongs to user)
  const vocabulary = await prisma.vocabulary.deleteMany({
    where: { id: vocabId, userId }
  })

  if (vocabulary.count === 0) {
    throw createError({
      statusCode: 404,
      message: 'Vocabulary not found'
    })
  }

  return { success: true }
})