import { prisma } from '../../utils/db'

export default defineEventHandler(async (event) => {
  const body = await readBody(event)

  if (!body.email) {
    throw createError({
      statusCode: 400,
      message: 'Email is required'
    })
  }

  const user = await prisma.user.findUnique({
    where: { email: body.email }
  })

  if (!user) {
    // Don't reveal if user exists or not
    return { hint: null }
  }

  if (!user.passwordHint) {
    return { hint: null }
  }

  return { hint: user.passwordHint }
})
