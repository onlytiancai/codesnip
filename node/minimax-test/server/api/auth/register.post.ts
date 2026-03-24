import bcrypt from 'bcryptjs'
import { prisma } from '../../utils/db'
import { validateCaptcha, getCaptchaFromSession } from '../../utils/captcha'

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const config = useRuntimeConfig()

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

  // CAPTCHA validation if enabled
  if (config.captchaEnabled === 'true') {
    const sessionId = getCaptchaFromSession(event)
    if (!body.captchaAnswer) {
      throw createError({
        statusCode: 400,
        message: 'CAPTCHA answer is required'
      })
    }
    if (!validateCaptcha(sessionId, body.captchaAnswer)) {
      throw createError({
        statusCode: 400,
        message: 'Invalid CAPTCHA answer'
      })
    }
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
      name: body.name || null,
      passwordHint: body.passwordHint || null
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
