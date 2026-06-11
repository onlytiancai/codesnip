import { prisma } from '../../utils/db'

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user) {
    throw createError({
      statusCode: 401,
      message: 'Not authenticated'
    })
  }

  const body = await readBody(event)

  if (!body.title || !body.content) {
    throw createError({
      statusCode: 400,
      message: 'Title and content are required'
    })
  }

  const article = await prisma.article.create({
    data: {
      title: body.title,
      content: body.content,
      description: body.description || null,
      url: body.url || null,
      userId: session.user.id,
      isPublished: body.isPublished || false,
      categoryId: body.categoryId || null
    },
    include: {
      category: true,
      tags: true,
      images: true
    }
  })

  if (body.tagIds && Array.isArray(body.tagIds)) {
    await prisma.article.update({
      where: { id: article.id },
      data: {
        tags: {
          connect: body.tagIds.map((id: string) => ({ id }))
        }
      }
    })
  }

  return { article }
})
