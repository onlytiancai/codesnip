import { prisma } from '../../utils/db'

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user) {
    throw createError({
      statusCode: 401,
      message: 'Not authenticated'
    })
  }

  const query = getQuery(event)
  const categoryId = query.categoryId as string | undefined
  const tagId = query.tagId as string | undefined

  const where: Record<string, unknown> = {
    userId: session.user.id
  }

  if (categoryId) {
    where.categoryId = categoryId
  }

  if (tagId) {
    where.tags = {
      some: {
        id: tagId
      }
    }
  }

  const articles = await prisma.article.findMany({
    where,
    include: {
      category: true,
      tags: true,
      images: true
    },
    orderBy: {
      createdAt: 'desc'
    }
  })

  return { articles }
})
