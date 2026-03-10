import { z } from 'zod'
import bcrypt from 'bcryptjs'
import { dbVerifyPassword } from '../../utils/db-auth'

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
    currentPassword: z.string().min(6),
    newPassword: z.string().min(6, 'Password must be at least 6 characters')
  })

  const parsed = schema.parse(body)

  // Get user
  const user = await prisma.user.findUnique({
    where: { id: userId }
  })

  if (!user || !user.password) {
    throw createError({
      statusCode: 400,
      message: 'Cannot change password for OAuth accounts'
    })
  }

  // Verify current password
  const isValid = await dbVerifyPassword(parsed.currentPassword, user.password)

  if (!isValid) {
    throw createError({
      statusCode: 400,
      message: 'Current password is incorrect'
    })
  }

  // Hash new password
  const hashedPassword = await bcrypt.hash(parsed.newPassword, 10)

  // Update password
  await prisma.user.update({
    where: { id: userId },
    data: { password: hashedPassword }
  })

  return { success: true, message: 'Password updated successfully' }
})