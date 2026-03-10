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
  const vocabId = parseInt(event.context.params?.id || '0')

  if (!vocabId) {
    throw createError({
      statusCode: 400,
      message: 'Vocabulary ID is required'
    })
  }

  const body = await readBody(event)

  // Validate request body
  const schema = z.object({
    progress: z.number().min(0).max(100).optional(),
    phonetic: z.string().max(50).optional(),
    definition: z.string().max(500).optional(),
    example: z.string().max(500).optional()
  })

  const parsed = schema.parse(body)

  // Check if vocabulary belongs to user
  const existing = await prisma.vocabulary.findFirst({
    where: { id: vocabId, userId }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: 'Vocabulary not found'
    })
  }

  // Update vocabulary
  const vocabulary = await prisma.vocabulary.update({
    where: { id: vocabId },
    data: {
      ...parsed,
      lastReviewAt: parsed.progress !== undefined ? new Date() : undefined
    }
  })

  return { vocabulary }
})