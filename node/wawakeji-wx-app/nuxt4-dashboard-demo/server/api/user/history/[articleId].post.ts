import { z } from 'zod'

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

  const body = await readBody(event)

  // Validate request body
  const schema = z.object({
    progress: z.number().min(0).max(100).optional()
  })

  let parsed
  try {
    parsed = schema.parse(body)
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw createError({
        statusCode: 400,
        message: error.errors?.[0]?.message || 'Validation failed'
      })
    }
    throw error
  }
  const progress = parsed.progress ?? 0

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

  // Upsert reading history
  const history = await prisma.readingHistory.upsert({
    where: {
      userId_articleId: {
        userId,
        articleId
      }
    },
    update: {
      progress,
      lastReadAt: new Date(),
      completedAt: progress === 100 ? new Date() : null
    },
    create: {
      userId,
      articleId,
      progress,
      completedAt: progress === 100 ? new Date() : null
    }
  })

  return { history }
})