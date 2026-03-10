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
  const body = await readBody(event)

  // Validate request body
  const schema = z.object({
    englishLevel: z.enum(['beginner', 'intermediate', 'advanced']).optional(),
    dailyGoal: z.number().int().min(5).max(60).optional(),
    audioSpeed: z.number().min(0.5).max(2.0).optional(),
    theme: z.enum(['light', 'dark', 'system']).optional(),
    fontSize: z.number().int().min(12).max(24).optional(),
    interests: z.array(z.string()).optional(),
    reminderEnabled: z.boolean().optional(),
    newArticleNotify: z.boolean().optional(),
    vocabReviewNotify: z.boolean().optional(),
    marketingEmails: z.boolean().optional()
  })

  const parsed = schema.parse(body)

  // Prepare update data
  const updateData: any = { ...parsed }
  if (parsed.interests) {
    updateData.interests = JSON.stringify(parsed.interests)
  }

  // Upsert preferences
  const preferences = await prisma.userPreferences.upsert({
    where: { userId },
    update: updateData,
    create: {
      userId,
      ...updateData
    }
  })

  return { preferences }
})