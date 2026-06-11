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
  const articleId = parseInt(event.context.params?.articleId || '0')

  if (!articleId) {
    throw createError({
      statusCode: 400,
      message: 'Article ID is required'
    })
  }

  // Delete bookmark
  const bookmark = await prisma.bookmark.delete({
    where: {
      userId_articleId: {
        userId,
        articleId
      }
    }
  }).catch(() => null)

  if (bookmark) {
    // Update article bookmark count
    await prisma.article.update({
      where: { id: articleId },
      data: { bookmarks: { decrement: 1 } }
    })
  }

  return { success: true }
})