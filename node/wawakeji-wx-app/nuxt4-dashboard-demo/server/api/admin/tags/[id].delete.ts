
export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid tag ID'
    })
  }

  // Check if tag exists
  const existing = await prisma.tag.findUnique({
    where: { id }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: 'Tag not found'
    })
  }

  // Delete tag (ArticleTag relations will be cascade deleted)
  await prisma.tag.delete({
    where: { id }
  })

  return { success: true }
})