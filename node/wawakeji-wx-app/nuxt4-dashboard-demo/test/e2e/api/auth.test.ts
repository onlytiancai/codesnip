import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsAdmin,
  createTestUser,
  createTestCategory,
  createTestTag,
  createTestArticle,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('Auth API', () => {
  describe('POST /api/auth/register', () => {
    it('should register a new user with valid data', async () => {
      const response = await apiRequest('/api/auth/register', {
        method: 'POST',
        body: {
          email: 'newuser@example.com',
          password: 'password123',
          confirmPassword: 'password123',
          name: 'New User',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.user).toBeDefined()
      expect(response.data.user.email).toBe('newuser@example.com')
      expect(response.data.user.name).toBe('New User')
      expect(response.data.user.id).toBeDefined()
    })

    it('should return 409 for existing email', async () => {
      // First registration
      await apiRequest('/api/auth/register', {
        method: 'POST',
        body: {
          email: 'existing@example.com',
          password: 'password123',
          confirmPassword: 'password123',
          name: 'Existing User',
        },
      })

      // Second registration with same email
      const response = await apiRequest('/api/auth/register', {
        method: 'POST',
        body: {
          email: 'existing@example.com',
          password: 'password123',
          confirmPassword: 'password123',
          name: 'Another User',
        },
      })

      expect(response.status).toBe(409)
      expect(response.data.message).toBe('Email already registered')
    })

    it('should return 400 for invalid email', async () => {
      const response = await apiRequest('/api/auth/register', {
        method: 'POST',
        body: {
          email: 'invalid-email',
          password: 'password123',
          confirmPassword: 'password123',
          name: 'Test User',
        },
      })

      expect(response.status).toBe(400)
    })

    it('should return 400 for short password', async () => {
      const response = await apiRequest('/api/auth/register', {
        method: 'POST',
        body: {
          email: 'shortpass@example.com',
          password: '12345',
          confirmPassword: '12345',
          name: 'Test User',
        },
      })

      expect(response.status).toBe(400)
    })

    it('should return 400 for mismatched passwords', async () => {
      const response = await apiRequest('/api/auth/register', {
        method: 'POST',
        body: {
          email: 'mismatch@example.com',
          password: 'password123',
          confirmPassword: 'password456',
          name: 'Test User',
        },
      })

      expect(response.status).toBe(400)
      expect(response.data.message).toContain('Validation')
    })

    it('should return 400 for short name', async () => {
      const response = await apiRequest('/api/auth/register', {
        method: 'POST',
        body: {
          email: 'shortname@example.com',
          password: 'password123',
          confirmPassword: 'password123',
          name: 'A',
        },
      })

      expect(response.status).toBe(400)
    })
  })

  describe('POST /api/auth/login', () => {
    it('should login with valid credentials', async () => {
      // Create user first
      await createTestUser({
        email: 'login-test@example.com',
        password: 'password123',
      })

      const response = await apiRequest('/api/auth/login', {
        method: 'POST',
        body: {
          email: 'login-test@example.com',
          password: 'password123',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.user).toBeDefined()
      expect(response.data.user.email).toBe('login-test@example.com')
    })

    it('should return 401 for invalid email', async () => {
      const response = await apiRequest('/api/auth/login', {
        method: 'POST',
        body: {
          email: 'nonexistent@example.com',
          password: 'password123',
        },
      })

      expect(response.status).toBe(401)
      expect(response.data.message).toBe('Invalid email or password')
    })

    it('should return 401 for wrong password', async () => {
      await createTestUser({
        email: 'wrongpass@example.com',
        password: 'correctpassword',
      })

      const response = await apiRequest('/api/auth/login', {
        method: 'POST',
        body: {
          email: 'wrongpass@example.com',
          password: 'wrongpassword',
        },
      })

      expect(response.status).toBe(401)
      expect(response.data.message).toBe('Invalid email or password')
    })

    it('should return 400 for missing fields', async () => {
      const response = await apiRequest('/api/auth/login', {
        method: 'POST',
        body: {
          email: 'test@example.com',
        },
      })

      expect(response.status).toBe(400)
    })
  })

  describe('POST /api/auth/logout', () => {
    it('should logout successfully', async () => {
      await loginAsAdmin()

      const response = await apiRequest('/api/auth/logout', {
        method: 'POST',
      })

      expect(response.status).toBe(200)
      expect(response.data.success).toBe(true)
    })
  })
})