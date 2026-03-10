import { z } from 'zod'

export default defineEventHandler(async (event) => {
  try {
    // Get current user from session
    const session = await getUserSession(event)

    if (!session?.user?.id) {
      throw createError({
        statusCode: 401,
        message: 'Unauthorized'
      })
    }

    const userId = session.user.id
    const body = await readBody(event)

    // Validate request body
    const schema = z.object({
      word: z.string().min(1, 'Word is required').max(100),
      phonetic: z.string().max(50).optional(),
      definition: z.string().min(1, 'Definition is required').max(500),
      example: z.string().max(500).optional(),
      articleId: z.number().int().optional()
    })

    const parsed = schema.parse(body)

    // Check if word already exists for this user
    const existing = await prisma.vocabulary.findUnique({
      where: {
        userId_word: {
          userId,
          word: parsed.word.toLowerCase()
        }
      }
    })

    if (existing) {
      throw createError({
        statusCode: 400,
        message: 'Word already exists in your vocabulary'
      })
    }

    // Create vocabulary entry
    const vocabulary = await prisma.vocabulary.create({
      data: {
        userId,
        word: parsed.word.toLowerCase(),
        phonetic: parsed.phonetic,
        definition: parsed.definition,
        example: parsed.example,
        articleId: parsed.articleId,
        progress: 0
      }
    })

    return { vocabulary }
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      const firstError = error.errors?.[0]?.message || 'Validation failed'
      throw createError({
        statusCode: 400,
        message: firstError
      })
    }
    throw error
  }
})