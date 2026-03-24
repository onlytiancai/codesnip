import bcrypt from 'bcryptjs'
import { prisma } from '../../utils/db'

export default defineEventHandler(async (event) => {
  const body = await readBody(event)

  if (!body.email || !body.password) {
    throw createError({
      statusCode: 400,
      message: 'Email and password are required'
    })
  }

  if (body.password.length < 6) {
    throw createError({
      statusCode: 400,
      message: 'Password must be at least 6 characters'
    })
  }

  const existingUser = await prisma.user.findUnique({
    where: { email: body.email }
  })

  if (existingUser) {
    throw createError({
      statusCode: 400,
      message: 'User already exists'
    })
  }

  const hashedPassword = await bcrypt.hash(body.password, 10)

  const user = await prisma.user.create({
    data: {
      email: body.email,
      password: hashedPassword,
      name: body.name || null
    }
  })

  await setUserSession(event, {
    user: {
      id: user.id,
      email: user.email,
      name: user.name
    }
  })

  return {
    user: {
      id: user.id,
      email: user.email,
      name: user.name
    }
  }
})
