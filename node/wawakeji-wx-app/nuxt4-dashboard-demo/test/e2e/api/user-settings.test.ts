import { describe, it, expect, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsTestUser,
  createTestUser,
  createTestUserPreferences,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('User Settings API', () => {
  beforeEach(async () => {
    await cleanupDatabase()
  })

  describe('GET /api/user/settings', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/settings')

      expect(response.status).toBe(401)
    })

    it('should return default preferences for new user', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/settings')

      expect(response.status).toBe(200)
      expect(response.data.englishLevel).toBeDefined()
      expect(response.data.dailyGoal).toBeDefined()
      expect(response.data.theme).toBeDefined()
    })

    it('should return existing preferences', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })

      await createTestUserPreferences(user!.id, {
        englishLevel: 'advanced',
        dailyGoal: 30,
        theme: 'dark',
      })

      const response = await apiRequest('/api/user/settings')

      expect(response.status).toBe(200)
      expect(response.data.englishLevel).toBe('advanced')
      expect(response.data.dailyGoal).toBe(30)
      expect(response.data.theme).toBe('dark')
    })
  })

  describe('PUT /api/user/settings', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/settings', {
        method: 'PUT',
        body: { englishLevel: 'advanced' },
      })

      expect(response.status).toBe(401)
    })

    it('should update preferences', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/settings', {
        method: 'PUT',
        body: {
          englishLevel: 'advanced',
          dailyGoal: 30,
          theme: 'dark',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.preferences.englishLevel).toBe('advanced')
      expect(response.data.preferences.dailyGoal).toBe(30)
      expect(response.data.preferences.theme).toBe('dark')
    })

    it('should update interests', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/settings', {
        method: 'PUT',
        body: {
          interests: ['technology', 'science'],
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.preferences.interests).toEqual(['technology', 'science'])
    })

    it('should update notification settings', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/settings', {
        method: 'PUT',
        body: {
          reminderEnabled: false,
          newArticleNotify: false,
          vocabReviewNotify: true,
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.preferences.reminderEnabled).toBe(false)
      expect(response.data.preferences.newArticleNotify).toBe(false)
      expect(response.data.preferences.vocabReviewNotify).toBe(true)
    })

    it('should return 400 for invalid english level', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/settings', {
        method: 'PUT',
        body: {
          englishLevel: 'invalid-level',
        },
      })

      expect(response.status).toBe(400)
    })
  })

  describe('PUT /api/user/password', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/password', {
        method: 'PUT',
        body: {
          currentPassword: 'oldpassword',
          newPassword: 'newpassword',
        },
      })

      expect(response.status).toBe(401)
    })

    it('should change password with valid credentials', async () => {
      // Create a new user for this test to avoid modifying the seed user's password
      const user = await createTestUser({
        email: 'password-change@example.com',
        password: 'password123',
      })

      // Login as this user
      await apiRequest('/api/auth/login', {
        method: 'POST',
        body: {
          email: 'password-change@example.com',
          password: 'password123',
        },
      })

      const response = await apiRequest('/api/user/password', {
        method: 'PUT',
        body: {
          currentPassword: 'password123',
          newPassword: 'newpassword123',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.success).toBe(true)

      // Verify new password works
      const loginResponse = await apiRequest('/api/auth/login', {
        method: 'POST',
        body: {
          email: 'password-change@example.com',
          password: 'newpassword123',
        },
      })
      expect(loginResponse.status).toBe(200)
    })

    it('should return 400 for wrong current password', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/password', {
        method: 'PUT',
        body: {
          currentPassword: 'wrongpassword',
          newPassword: 'newpassword123',
        },
      })

      expect(response.status).toBe(400)
      expect(response.data.message).toContain('incorrect')
    })

    it('should return 400 for short new password', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/password', {
        method: 'PUT',
        body: {
          currentPassword: 'user123',
          newPassword: '12345', // Too short
        },
      })

      expect(response.status).toBe(400)
    })
  })

  describe('DELETE /api/user/account', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/account', {
        method: 'DELETE',
        body: { confirm: 'DELETE MY ACCOUNT' },
      })

      expect(response.status).toBe(401)
    })

    it('should delete account with correct confirmation', async () => {
      // Create a new user to delete (not the seed user)
      const user = await createTestUser({
        email: 'delete-test@example.com',
        password: 'password123',
      })

      // Login as this user
      await apiRequest('/api/auth/login', {
        method: 'POST',
        body: {
          email: 'delete-test@example.com',
          password: 'password123',
        },
      })

      const response = await apiRequest('/api/user/account', {
        method: 'DELETE',
        body: { confirm: 'DELETE MY ACCOUNT' },
      })

      expect(response.status).toBe(200)
      expect(response.data.success).toBe(true)

      // Verify user is deleted
      const deletedUser = await prisma.user.findUnique({
        where: { id: user.id },
      })
      expect(deletedUser).toBeNull()
    })

    it('should return 400 for wrong confirmation', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/account', {
        method: 'DELETE',
        body: { confirm: 'WRONG CONFIRMATION' },
      })

      expect(response.status).toBe(400)
    })
  })
})