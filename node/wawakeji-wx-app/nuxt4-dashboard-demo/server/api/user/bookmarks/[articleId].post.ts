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

  // Check if article exists
  const article = await prisma.article.findUnique({
    where: { id: articleId }
  })

  if (!article) {
    throw createError({
      statusCode: 404,
      message: 'Article not found'
    })
  }

  // Check if already bookmarked
  const existing = await prisma.bookmark.findUnique({
    where: {
      userId_articleId: {
        userId,
        articleId
      }
    }
  })

  if (existing) {
    return { bookmark: existing, message: 'Already bookmarked' }
  }

  // Create bookmark
  const bookmark = await prisma.bookmark.create({
    data: {
      userId,
      articleId
    }
  })

  // Update article bookmark count
  await prisma.article.update({
    where: { id: articleId },
    data: { bookmarks: { increment: 1 } }
  })

  return { bookmark }
})