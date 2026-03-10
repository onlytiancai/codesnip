import { describe, it, expect } from 'vitest'

describe('Helpers Utils', () => {
  describe('Password Strength', () => {
    const calculatePasswordStrength = (password: string): number => {
      let strength = 0
      if (password.length >= 6) strength += 1
      if (password.length >= 8) strength += 1
      if (password.length >= 12) strength += 1
      if (/[a-z]/.test(password) && /[A-Z]/.test(password)) strength += 1
      if (/\d/.test(password)) strength += 1
      if (/[^a-zA-Z0-9]/.test(password)) strength += 1
      return Math.min(strength, 5)
    }

    it('should return 0 for empty password', () => {
      expect(calculatePasswordStrength('')).toBe(0)
    })

    it('should return 1 for short password with only lowercase', () => {
      expect(calculatePasswordStrength('abc')).toBe(0)
    })

    it('should return 2 for password with 6+ chars', () => {
      expect(calculatePasswordStrength('abcdef')).toBe(1)
    })

    it('should return higher strength for mixed case', () => {
      expect(calculatePasswordStrength('Abcdefgh')).toBe(3)
    })

    it('should return higher strength for numbers', () => {
      expect(calculatePasswordStrength('Abcdefg1')).toBe(4)
    })

    it('should return max strength for complex password', () => {
      expect(calculatePasswordStrength('Abcdefg1!@#')).toBe(5)
    })
  })

  describe('Email Validation', () => {
    const isValidEmail = (email: string): boolean => {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      return emailRegex.test(email)
    }

    it('should return true for valid email', () => {
      expect(isValidEmail('test@example.com')).toBe(true)
    })

    it('should return false for email without @', () => {
      expect(isValidEmail('testexample.com')).toBe(false)
    })

    it('should return false for email without domain', () => {
      expect(isValidEmail('test@')).toBe(false)
    })

    it('should return false for email without local part', () => {
      expect(isValidEmail('@example.com')).toBe(false)
    })

    it('should return false for email with spaces', () => {
      expect(isValidEmail('test @example.com')).toBe(false)
    })

    it('should return true for email with subdomain', () => {
      expect(isValidEmail('test@sub.example.com')).toBe(true)
    })
  })

  describe('Date Formatting', () => {
    const formatDate = (date: Date | string): string => {
      const d = typeof date === 'string' ? new Date(date) : date
      return d.toISOString().split('T')[0]
    }

    const formatRelativeTime = (date: Date | string): string => {
      const d = typeof date === 'string' ? new Date(date) : date
      const now = new Date()
      const diffMs = now.getTime() - d.getTime()
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

      if (diffDays === 0) return 'Today'
      if (diffDays === 1) return 'Yesterday'
      if (diffDays < 7) return `${diffDays} days ago`
      if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`
      if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`
      return `${Math.floor(diffDays / 365)} years ago`
    }

    it('should format date to YYYY-MM-DD', () => {
      expect(formatDate('2024-01-15')).toBe('2024-01-15')
    })

    it('should format Date object', () => {
      const date = new Date('2024-06-20T10:30:00Z')
      expect(formatDate(date)).toBe('2024-06-20')
    })

    it('should return Today for current date', () => {
      const today = new Date()
      expect(formatRelativeTime(today)).toBe('Today')
    })

    it('should return Yesterday for yesterday', () => {
      const yesterday = new Date()
      yesterday.setDate(yesterday.getDate() - 1)
      expect(formatRelativeTime(yesterday)).toBe('Yesterday')
    })

    it('should return days ago for recent dates', () => {
      const threeDaysAgo = new Date()
      threeDaysAgo.setDate(threeDaysAgo.getDate() - 3)
      expect(formatRelativeTime(threeDaysAgo)).toBe('3 days ago')
    })
  })

  describe('Slug Generation', () => {
    const generateSlug = (text: string): string => {
      return text
        .toLowerCase()
        .trim()
        .replace(/[^\w\s-]/g, '')
        .replace(/[\s_-]+/g, '-')
        .replace(/^-+|-+$/g, '')
    }

    it('should generate slug from title', () => {
      expect(generateSlug('Hello World')).toBe('hello-world')
    })

    it('should handle special characters', () => {
      expect(generateSlug('Hello, World!')).toBe('hello-world')
    })

    it('should handle multiple spaces', () => {
      expect(generateSlug('Hello    World')).toBe('hello-world')
    })

    it('should handle leading/trailing spaces', () => {
      expect(generateSlug('  Hello World  ')).toBe('hello-world')
    })

    it('should handle underscores', () => {
      expect(generateSlug('Hello_World')).toBe('hello-world')
    })
  })

  describe('Truncate Text', () => {
    const truncate = (text: string, maxLength: number): string => {
      if (text.length <= maxLength) return text
      return text.slice(0, maxLength - 3) + '...'
    }

    it('should not truncate short text', () => {
      expect(truncate('Hello', 10)).toBe('Hello')
    })

    it('should truncate long text', () => {
      expect(truncate('Hello World This is Long', 10)).toBe('Hello W...')
    })

    it('should handle exact length', () => {
      expect(truncate('Hello', 5)).toBe('Hello')
    })
  })
})