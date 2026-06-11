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
      name: z.string().min(1, 'Name is required').max(100).optional(),
      avatar: z.string().url().optional().nullable(),
      bio: z.string().max(500).optional().nullable()
    })

    const parsed = schema.parse(body)

    // Update user
    const user = await prisma.user.update({
      where: { id: userId },
      data: parsed,
      select: {
        id: true,
        email: true,
        name: true,
        avatar: true,
        bio: true,
        role: true
      }
    })

    // Update session
    await setUserSession(event, {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        avatar: user.avatar,
        role: user.role
      }
    })

    return { user }
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