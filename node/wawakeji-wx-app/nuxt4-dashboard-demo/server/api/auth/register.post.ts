import { z } from 'zod'
import { dbHashPassword, findUserByEmail, createUser } from '../../utils/db-auth'

export default defineEventHandler(async (event) => {
  try {
    const body = await readBody(event)

    // Validate request body
    const schema = z.object({
      name: z.string().min(2, 'Name must be at least 2 characters'),
      email: z.string().email('Please enter a valid email address'),
      password: z.string().min(6, 'Password must be at least 6 characters'),
      confirmPassword: z.string(),
    }).refine((data) => data.password === data.confirmPassword, {
      message: 'Passwords do not match',
      path: ['confirmPassword'],
    })

    const parsed = schema.parse(body)

    // Check if email already exists
    const existingUser = await findUserByEmail(parsed.email)

    if (existingUser) {
      throw createError({
        statusCode: 409,
        message: 'Email already registered',
      })
    }

    // Hash password
    const hashedPassword = await dbHashPassword(parsed.password)

    // Create user
    const user = await createUser({
      email: parsed.email,
      name: parsed.name,
      password: hashedPassword,
    })

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