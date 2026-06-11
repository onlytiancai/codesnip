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

  const user = await prisma.user.findUnique({
    where: { email: body.email }
  })

  if (!user) {
    throw createError({
      statusCode: 401,
      message: 'Invalid credentials'
    })
  }

  if (user.isDisabled) {
    throw createError({
      statusCode: 403,
      message: 'Account is disabled'
    })
  }

  const isValidPassword = await bcrypt.compare(body.password, user.password)

  if (!isValidPassword) {
    throw createError({
      statusCode: 401,
      message: 'Invalid credentials'
    })
  }

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
