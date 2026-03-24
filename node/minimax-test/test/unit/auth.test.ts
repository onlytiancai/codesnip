import { describe, it, expect } from 'vitest'
import bcrypt from 'bcryptjs'

describe('Auth', () => {
  it('should hash and verify password correctly', async () => {
    const password = 'testPassword123'
    const hashed = await bcrypt.hash(password, 10)

    expect(hashed).not.toBe(password)
    expect(hashed.length).toBeGreaterThan(20)

    const isValid = await bcrypt.compare(password, hashed)
    expect(isValid).toBe(true)
  })

  it('should reject incorrect password', async () => {
    const password = 'testPassword123'
    const hashed = await bcrypt.hash(password, 10)

    const isValid = await bcrypt.compare('wrongPassword', hashed)
    expect(isValid).toBe(false)
  })
})
