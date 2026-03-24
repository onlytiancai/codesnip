import bcrypt from 'bcryptjs'
import { prisma } from '../../../../utils/db'

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user || session.user.id !== '1') {
    throw createError({ statusCode: 403, message: 'Admin only' })
  }

  const userId = getRouterParam(event, 'id')
  const body = await readBody(event)

  if (!body.newPassword || body.newPassword.length < 6) {
    throw createError({ statusCode: 400, message: 'Password must be at least 6 characters' })
  }

  const hashedPassword = await bcrypt.hash(body.newPassword, 10)

  const user = await prisma.user.update({
    where: { id: userId },
    data: { password: hashedPassword }
  })

  return {
    success: true,
    user: {
      id: user.id,
      email: user.email,
      name: user.name
    }
  }
})
