import { z } from 'zod'

const userSchema = z.object({
  name: z.string().optional(),
  avatar: z.string().optional().nullable(),
  role: z.enum(['USER', 'ADMIN']).optional()
})

export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')
  const body = await readBody(event)

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid user ID'
    })
  }

  // Validate input
  const data = userSchema.parse(body)

  // Check if user exists
  const existing = await prisma.user.findUnique({
    where: { id }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: 'User not found'
    })
  }

  // Update user
  const user = await prisma.user.update({
    where: { id },
    data: {
      ...data,
      updatedAt: new Date()
    },
    select: {
      id: true,
      email: true,
      name: true,
      avatar: true,
      role: true,
      createdAt: true,
      updatedAt: true
    }
  })

  return user
})