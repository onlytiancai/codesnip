
export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid article ID'
    })
  }

  // Check if article exists
  const existing = await prisma.article.findUnique({
    where: { id }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: 'Article not found'
    })
  }

  // Delete article (sentences and tags will be cascade deleted)
  await prisma.article.delete({
    where: { id }
  })

  return { success: true }
})