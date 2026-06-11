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

    // Validate confirmation
    const schema = z.object({
      confirm: z.literal('DELETE MY ACCOUNT', {
        errorMap: () => ({ message: 'Please type DELETE MY ACCOUNT to confirm' })
      })
    })

    schema.parse(body)

    // Delete user (cascade will delete all related data)
    await prisma.user.delete({
      where: { id: userId }
    })

    // Clear session
    await clearUserSession(event)

    return { success: true, message: 'Account deleted successfully' }
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