import { createHash } from 'crypto'

interface CaptchaData {
  answer: string
  expiresAt: number
}

const captchaStore = new Map<string, CaptchaData>()

export function generateCaptcha(sessionId: string): { question: string; captchaId: string } {
  const num1 = Math.floor(Math.random() * 10) + 1
  const num2 = Math.floor(Math.random() * 10) + 1
  const answer = (num1 + num2).toString()
  const expiresAt = Date.now() + 5 * 60 * 1000 // 5 minutes

  captchaStore.set(sessionId, { answer, expiresAt })

  return {
    question: `${num1} + ${num2} = ?`,
    captchaId: sessionId
  }
}

export function validateCaptcha(sessionId: string, userAnswer: string): boolean {
  const captchaData = captchaStore.get(sessionId)

  if (!captchaData) {
    return false
  }

  if (Date.now() > captchaData.expiresAt) {
    captchaStore.delete(sessionId)
    return false
  }

  const isValid = captchaData.answer === userAnswer.trim()
  if (isValid) {
    captchaStore.delete(sessionId)
  }

  return isValid
}

export function getCaptchaFromSession(event: any): string {
  const session = event.context.session || event.context.user
  return session?.id || event.context.cookies?.session?.id || 'anonymous'
}
