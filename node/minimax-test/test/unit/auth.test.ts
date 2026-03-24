import { describe, it, expect } from 'vitest'
import { hashPassword, verifyPassword } from '../../server/utils/auth'

describe('Auth', () => {
  it('should hash and verify password correctly', async () => {
    const password = 'testPassword123'
    const hashed = await hashPassword(password)

    expect(hashed).not.toBe(password)
    expect(hashed.length).toBeGreaterThan(20)

    const isValid = await verifyPassword(password, hashed)
    expect(isValid).toBe(true)
  })

  it('should reject incorrect password', async () => {
    const password = 'testPassword123'
    const hashed = await hashPassword(password)

    const isValid = await verifyPassword('wrongPassword', hashed)
    expect(isValid).toBe(false)
  })
})
