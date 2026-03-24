import { generateCaptcha, getCaptchaFromSession } from '../../utils/captcha'

export default defineEventHandler(async (event) => {
  const sessionId = getCaptchaFromSession(event)
  const captcha = generateCaptcha(sessionId)

  return {
    question: captcha.question
  }
})
