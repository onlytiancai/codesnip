import { z } from 'zod'
import { dbVerifyPassword, findUserByEmail } from '../../utils/db-auth'

export default defineEventHandler(async (event) => {
  try {
    const body = await readBody(event)

    // Validate request body
    const schema = z.object({
      email: z.string().email('Please enter a valid email address'),
      password: z.string().min(6, 'Password must be at least 6 characters'),
    })

    const parsed = schema.parse(body)

    // Find user
    const user = await findUserByEmail(parsed.email)

    if (!user || !user.password) {
      throw createError({
        statusCode: 401,
        message: 'Invalid email or password',
      })
    }

    // Verify password
    const isValid = await dbVerifyPassword(parsed.password, user.password)

    if (!isValid) {
      throw createError({
        statusCode: 401,
        message: 'Invalid email or password',
      })
    }

    // Set user session
    await setUserSession(event, {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        avatar: user.avatar,
        role: user.role,
        hasPassword: !!user.password,
      },
    })

    return {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        avatar: user.avatar,
        role: user.role,
      },
    }
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      const firstError = error.errors?.[0]?.message || 'Validation failed'
      throw createError({
        statusCode: 400,
        message: firstError,
      })
    }
    throw error
  }
})