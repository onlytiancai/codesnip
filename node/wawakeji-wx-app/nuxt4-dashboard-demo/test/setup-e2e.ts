import { beforeAll, afterAll, beforeEach } from 'vitest'
import { $fetch, FetchOptions } from 'ofetch'
import { PrismaBetterSQLite3 } from '@prisma/adapter-better-sqlite3'
import { PrismaClient } from '../generated/prisma/client'
import bcrypt from 'bcryptjs'
import 'dotenv/config'
import http from 'http'

// Database connection for tests
const adapter = new PrismaBetterSQLite3({ url: process.env.DATABASE_URL! })
const prisma = new PrismaClient({ adapter })

// Test server URL
const baseURL = process.env.NUXT_PUBLIC_BASE_URL || 'http://localhost:3000'

// Cookie store for the current test context
let currentCookies: string | undefined

// Helper to clear cookies (for testing unauthenticated requests)
export function clearAuthCookies() {
  currentCookies = undefined
}

// Helper to make API requests using native http module for proper cookie handling
export async function apiRequest(
  path: string,
  options: Omit<RequestInit, 'body'> & { body?: Record<string, unknown> } = {}
): Promise<{ status: number; data: any; headers: Headers }> {
  return new Promise((resolve, reject) => {
    const url = new URL(`${baseURL}${path}`)

    const requestOptions: http.RequestOptions = {
      hostname: url.hostname,
      port: url.port || 3000,
      path: url.pathname + url.search,
      method: (options.method as string) || 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(currentCookies ? { 'Cookie': currentCookies } : {})
      }
    }

    const bodyData = options.body ? JSON.stringify(options.body) : null
    if (bodyData) {
      requestOptions.headers!['Content-Length'] = Buffer.byteLength(bodyData)
    }

    const req = http.request(requestOptions, (res) => {
      // Capture set-cookie header
      const setCookie = res.headers['set-cookie']
      if (setCookie) {
        currentCookies = setCookie.join('; ')
      }

      let data = ''
      res.on('data', (chunk) => { data += chunk })
      res.on('end', () => {
        try {
          const parsedData = JSON.parse(data)
          resolve({
            status: res.statusCode || 0,
            data: parsedData,
            headers: res.headers as any
          })
        } catch {
          resolve({
            status: res.statusCode || 0,
            data: null,
            headers: res.headers as any
          })
        }
      })
    })

    req.on('error', reject)

    if (bodyData) {
      req.write(bodyData)
    }
    req.end()
  })
}

// Helper to create test user
export async function createTestUser(overrides: Partial<{
  email: string
  password: string
  name: string
  role: string
}> = {}) {
  const password = overrides.password || 'password123'
  const hashedPassword = await bcrypt.hash(password, 10)

  const user = await prisma.user.create({
    data: {
      email: overrides.email || `test-${Date.now()}@example.com`,
      name: overrides.name || 'Test User',
      password: hashedPassword,
      role: overrides.role || 'USER',
    },
  })

  return { ...user, plainPassword: password }
}

// Helper to create test admin
export async function createTestAdmin(overrides: Partial<{
  email: string
  password: string
  name: string
}> = {}) {
  return createTestUser({ ...overrides, role: 'ADMIN' })
}

// Helper to login and get session
export async function loginAsUser(email: string, password: string) {
  const result = await apiRequest('/api/auth/login', {
    method: 'POST',
    body: { email, password },
  })
  return result
}

// Helper to login as admin
export async function loginAsAdmin() {
  const admin = await prisma.user.findUnique({
    where: { email: 'admin@example.com' },
  })

  if (!admin) {
    throw new Error('Admin user not found. Run db:seed first.')
  }

  return loginAsUser('admin@example.com', 'admin123')
}

// Helper to create test category
export async function createTestCategory(overrides: Partial<{
  name: string
  slug: string
  description: string
  color: string
}> = {}) {
  const slug = overrides.slug || `test-category-${Date.now()}`

  return prisma.category.create({
    data: {
      name: overrides.name || slug,
      slug,
      description: overrides.description || 'Test category',
      color: overrides.color || '#3b82f6',
    },
  })
}

// Helper to create test tag
export async function createTestTag(overrides: Partial<{
  name: string
  slug: string
  description: string
  color: string
}> = {}) {
  const slug = overrides.slug || `test-tag-${Date.now()}`

  return prisma.tag.create({
    data: {
      name: overrides.name || slug,
      slug,
      description: overrides.description || 'Test tag',
      color: overrides.color || '#3b82f6',
    },
  })
}

// Helper to create test article
export async function createTestArticle(authorId: number, overrides: Partial<{
  title: string
  slug: string
  excerpt: string
  content: string
  status: string
  difficulty: string
  categoryId: number
}> = {}) {
  const slug = overrides.slug || `test-article-${Date.now()}`

  return prisma.article.create({
    data: {
      title: overrides.title || 'Test Article',
      slug,
      excerpt: overrides.excerpt || 'Test excerpt',
      content: overrides.content || 'Test content',
      status: overrides.status || 'draft',
      difficulty: overrides.difficulty || 'beginner',
      categoryId: overrides.categoryId || null,
      authorId,
    },
  })
}

// Helper to create test reading history
export async function createTestReadingHistory(userId: number, articleId: number, overrides: Partial<{
  progress: number
  completedAt: Date | null
}> = {}) {
  return prisma.readingHistory.create({
    data: {
      userId,
      articleId,
      progress: overrides.progress ?? 0,
      completedAt: overrides.completedAt ?? null,
    },
  })
}

// Helper to create test bookmark
export async function createTestBookmark(userId: number, articleId: number) {
  return prisma.bookmark.create({
    data: {
      userId,
      articleId,
    },
  })
}

// Helper to create test vocabulary
export async function createTestVocabulary(userId: number, overrides: Partial<{
  word: string
  phonetic: string
  definition: string
  example: string
  progress: number
  articleId: number
}> = {}) {
  return prisma.vocabulary.create({
    data: {
      userId,
      word: overrides.word || `test-word-${Date.now()}`,
      phonetic: overrides.phonetic || '/test/',
      definition: overrides.definition || 'Test definition',
      example: overrides.example || 'Test example sentence.',
      progress: overrides.progress ?? 0,
      articleId: overrides.articleId || null,
    },
  })
}

// Helper to create test user preferences
export async function createTestUserPreferences(userId: number, overrides: Partial<{
  englishLevel: string
  dailyGoal: number
  audioSpeed: number
  theme: string
  fontSize: number
  interests: string[]
  reminderEnabled: boolean
  newArticleNotify: boolean
  vocabReviewNotify: boolean
  marketingEmails: boolean
}> = {}) {
  return prisma.userPreferences.create({
    data: {
      userId,
      englishLevel: overrides.englishLevel || 'intermediate',
      dailyGoal: overrides.dailyGoal || 10,
      audioSpeed: overrides.audioSpeed || 1.0,
      theme: overrides.theme || 'system',
      fontSize: overrides.fontSize || 16,
      interests: overrides.interests ? JSON.stringify(overrides.interests) : null,
      reminderEnabled: overrides.reminderEnabled ?? true,
      newArticleNotify: overrides.newArticleNotify ?? true,
      vocabReviewNotify: overrides.vocabReviewNotify ?? false,
      marketingEmails: overrides.marketingEmails ?? false,
    },
  })
}

// Helper to create test membership
export async function createTestMembership(userId: number, overrides: Partial<{
  plan: string
  endDate: Date | null
}> = {}) {
  return prisma.membership.create({
    data: {
      userId,
      plan: overrides.plan || 'free',
      endDate: overrides.endDate || null,
    },
  })
}

// Helper to login as regular user
export async function loginAsTestUser() {
  const user = await prisma.user.findUnique({
    where: { email: 'user@example.com' },
  })

  if (!user) {
    throw new Error('Test user not found. Run db:seed first.')
  }

  return loginAsUser('user@example.com', 'user123')
}

// Clean up database helper
export async function cleanupDatabase() {
  await prisma.sentence.deleteMany()
  await prisma.articleTag.deleteMany()
  await prisma.article.deleteMany()
  await prisma.tag.deleteMany()
  await prisma.category.deleteMany()
  await prisma.account.deleteMany()

  // Delete user-related data first (due to foreign key constraints)
  await prisma.vocabulary.deleteMany()
  await prisma.bookmark.deleteMany()
  await prisma.readingHistory.deleteMany()
  await prisma.userPreferences.deleteMany()
  await prisma.membership.deleteMany()

  // Don't delete seed users, just test users
  await prisma.user.deleteMany({
    where: {
      email: { notIn: ['admin@example.com', 'user@example.com'] },
    },
  })

  // Clear stored cookies for this test context
  currentCookies = undefined
}

// Setup and teardown
beforeAll(async () => {
  // Ensure database is seeded
  const admin = await prisma.user.findUnique({
    where: { email: 'admin@example.com' },
  })

  if (!admin) {
    throw new Error('Database not seeded. Run pnpm db:seed first.')
  }
})

afterAll(async () => {
  await prisma.$disconnect()
})

beforeEach(async () => {
  // Clean up test data before each test
  await cleanupDatabase()
})

// Export prisma for use in tests
export { prisma }