
export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid category ID'
    })
  }

  // Check if category exists
  const existing = await prisma.category.findUnique({
    where: { id },
    include: {
      _count: {
        select: { Article: true }
      }
    }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: 'Category not found'
    })
  }

  // Check if category has articles
  if (existing._count.Article > 0) {
    throw createError({
      statusCode: 400,
      message: `Cannot delete category with ${existing._count.Article} articles. Please reassign or delete the articles first.`
    })
  }

  // Delete category
  await prisma.category.delete({
    where: { id }
  })

  return { success: true }
})