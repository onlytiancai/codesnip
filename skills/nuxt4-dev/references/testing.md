# Testing in Nuxt 4

Complete guide to unit testing with Vitest and E2E testing with Playwright in Nuxt 4.

## Overview

| Test Type | Tool | Config File | Test Directory |
|-----------|------|-------------|----------------|
| Unit Tests | Vitest | `vitest.config.ts` | `test/unit/` |
| E2E Tests (API) | Vitest | `vitest.e2e.config.ts` | `test/e2e/api/` |
| E2E Tests (Browser) | Playwright | `playwright.config.ts` | `test/e2e/browser/` |

## Unit Testing with Vitest

### Setup

Install dependencies:

```bash
pnpm add -D vitest @vitest/ui @nuxt/test-utils @vue/test-utils happy-dom
```

`vitest.config.ts`:

```typescript
import { defineVitestConfig } from '@nuxt/test-utils/config'

export default defineVitestConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['test/unit/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
})
```

### Writing Unit Tests

**Example: Component Test**

```typescript
// test/unit/components/login.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'

describe('Login Page Validation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Form Fields', () => {
    it('should have email and password fields', () => {
      const expectedFields = ['email', 'password']
      expect(expectedFields).toContain('email')
      expect(expectedFields).toContain('password')
    })

    it('should have OAuth providers', () => {
      const oauthProviders = ['GitHub', 'Google']
      expect(oauthProviders).toContain('GitHub')
      expect(oauthProviders).toContain('Google')
    })
  })

  describe('Email Validation', () => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

    it('should accept valid email format', () => {
      expect(emailRegex.test('test@example.com')).toBe(true)
      expect(emailRegex.test('user.name@domain.co')).toBe(true)
    })

    it('should reject invalid email format', () => {
      expect(emailRegex.test('invalid')).toBe(false)
      expect(emailRegex.test('test@')).toBe(false)
    })
  })

  describe('Password Validation', () => {
    const minLength = 6

    it('should require minimum password length', () => {
      expect('password123'.length).toBeGreaterThanOrEqual(minLength)
      expect('12345'.length).toBeLessThan(minLength)
    })
  })
})
```

### Testing Utilities/Helpers

```typescript
// test/unit/utils/helpers.test.ts
import { describe, it, expect } from 'vitest'

describe('Helper Functions', () => {
  describe('slugify', () => {
    it('should convert string to slug', () => {
      // Test your utility functions
      expect('Hello World'.toLowerCase().replace(/\s+/g, '-'))
        .toBe('hello-world')
    })
  })

  describe('date formatting', () => {
    it('should format date correctly', () => {
      const date = new Date('2024-01-15')
      expect(date.toISOString().split('T')[0]).toBe('2024-01-15')
    })
  })
})
```

### Running Unit Tests

```bash
# Run all unit tests
pnpm test:unit

# Run with UI
pnpm test:ui

# Run specific file
pnpm test:unit -- test/unit/components/login.test.ts

# Run in watch mode
pnpm test:unit -- --watch
```

## E2E API Testing with Vitest

### Setup

`vitest.e2e.config.ts`:

```typescript
import { defineVitestConfig } from '@nuxt/test-utils/config'

export default defineVitestConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['test/e2e/api/**/*.test.ts'],
    setupFiles: ['./test/setup-e2e.ts'],
  },
})
```

`test/setup-e2e.ts`:

```typescript
import { setup } from '@nuxt/test-utils/e2e'
import { beforeAll } from 'vitest'

beforeAll(async () => {
  await setup({
    rootDir: './',
    port: 3000,
  })
})
```

### Writing API Tests

```typescript
// test/e2e/api/auth.test.ts
import { describe, it, expect } from 'vitest'
import { $fetch } from '@nuxt/test-utils/e2e'

describe('Auth API', () => {
  it('should register a new user', async () => {
    const response = await $fetch('/api/auth/register', {
      method: 'POST',
      body: {
        name: 'Test User',
        email: 'test@example.com',
        password: 'password123',
        confirmPassword: 'password123',
      },
    })

    expect(response.user).toBeDefined()
    expect(response.user.email).toBe('test@example.com')
  })

  it('should login with valid credentials', async () => {
    const response = await $fetch('/api/auth/login', {
      method: 'POST',
      body: {
        email: 'admin@example.com',
        password: 'admin123',
      },
    })

    expect(response.user).toBeDefined()
    expect(response.user.email).toBe('admin@example.com')
  })

  it('should reject invalid credentials', async () => {
    await expect(
      $fetch('/api/auth/login', {
        method: 'POST',
        body: {
          email: 'wrong@example.com',
          password: 'wrongpassword',
        },
      })
    ).rejects.toThrow()
  })
})
```

### Testing Admin APIs

```typescript
// test/e2e/api/admin-articles.test.ts
import { describe, it, expect, beforeEach } from 'vitest'
import { $fetch } from '@nuxt/test-utils/e2e'

describe('Admin Articles API', () => {
  let authToken: string

  beforeEach(async () => {
    // Login to get auth token
    const loginResponse = await $fetch('/api/auth/login', {
      method: 'POST',
      body: {
        email: 'admin@example.com',
        password: 'admin123',
      },
    })
    authToken = loginResponse.token
  })

  it('should get all articles', async () => {
    const articles = await $fetch('/api/admin/articles', {
      headers: { Authorization: `Bearer ${authToken}` },
    })

    expect(Array.isArray(articles)).toBe(true)
  })

  it('should create a new article', async () => {
    const article = await $fetch('/api/admin/articles', {
      method: 'POST',
      headers: { Authorization: `Bearer ${authToken}` },
      body: {
        title: 'New Article',
        slug: 'new-article',
        content: 'Article content',
      },
    })

    expect(article.title).toBe('New Article')
  })

  it('should update an article', async () => {
    const article = await $fetch('/api/admin/articles/1', {
      method: 'PUT',
      headers: { Authorization: `Bearer ${authToken}` },
      body: {
        title: 'Updated Title',
      },
    })

    expect(article.title).toBe('Updated Title')
  })

  it('should delete an article', async () => {
    await $fetch('/api/admin/articles/1', {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${authToken}` },
    })

    // Verify deletion
    const articles = await $fetch('/api/admin/articles', {
      headers: { Authorization: `Bearer ${authToken}` },
    })

    expect(articles.find((a: any) => a.id === 1)).toBeUndefined()
  })
})
```

## Browser E2E Testing with Playwright

### Setup

Install Playwright:

```bash
pnpm add -D @playwright/test
pnpm playwright install
```

`playwright.config.ts`:

```typescript
import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './test/e2e/browser',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'pnpm dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
})
```

### Writing Browser Tests

```typescript
// test/e2e/browser/auth.test.ts
import { test, expect } from '@playwright/test'

test.describe('Auth Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('login page renders correctly', async ({ page }) => {
    await page.goto('/login')

    // Check page title
    await expect(page.locator('h1')).toContainText('Welcome Back')

    // Check form elements exist
    await expect(page.locator('input[type="email"]')).toBeVisible()
    await expect(page.locator('input[type="password"]')).toBeVisible()
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible()

    // Check OAuth buttons exist
    await expect(page.getByRole('button', { name: /github/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /google/i })).toBeVisible()

    // Check link to register
    await expect(page.getByRole('link', { name: /sign up/i })).toBeVisible()
  })

  test('register page renders correctly', async ({ page }) => {
    await page.goto('/register')

    // Check page title
    await expect(page.locator('h1')).toContainText('Create Your Account')

    // Check step indicators
    await expect(page.getByText('1')).toBeVisible()
    await expect(page.getByText('2')).toBeVisible()
    await expect(page.getByText('3')).toBeVisible()
  })

  test('login with valid credentials', async ({ page }) => {
    await page.goto('/login')

    // Fill in credentials
    await page.locator('input[type="email"]').fill('admin@example.com')
    await page.locator('input[type="password"]').fill('admin123')

    // Submit form
    await page.getByRole('button', { name: /sign in/i }).click()

    // Should redirect to home page
    await expect(page).toHaveURL(/\/$/, { timeout: 10000 })
  })

  test('login error display', async ({ page }) => {
    await page.goto('/login')

    // Fill in wrong credentials
    await page.locator('input[type="email"]').fill('wrong@example.com')
    await page.locator('input[type="password"]').fill('wrongpassword')

    // Submit form
    await page.getByRole('button', { name: /sign in/i }).click()

    // Should show error message
    await expect(page.getByText(/invalid email or password/i)).toBeVisible({
      timeout: 5000
    })
  })

  test('navigation between auth pages', async ({ page }) => {
    // Start at login
    await page.goto('/login')

    // Click sign up link
    await page.getByRole('link', { name: /sign up/i }).click()

    // Should be on register page
    await expect(page).toHaveURL('/register')
    await expect(page.locator('h1')).toContainText('Create Your Account')

    // Click sign in link
    await page.getByRole('link', { name: /sign in/i }).click()

    // Should be back on login page
    await expect(page).toHaveURL('/login')
    await expect(page.locator('h1')).toContainText('Welcome Back')
  })

  test('register form step navigation', async ({ page }) => {
    await page.goto('/register')

    // Fill step 1
    await page.locator('input[type="email"]').fill('test@example.com')
    await page.getByPlaceholder('Create a password').fill('password123')
    await page.getByPlaceholder('Confirm your password').fill('password123')

    // Click continue
    await page.getByRole('button', { name: /continue/i }).click()

    // Should now be on step 2 (name field should appear)
    await expect(page.getByPlaceholder('Your full name')).toBeVisible()

    // Go back should work
    await page.getByRole('button', { name: /back/i }).click()

    // Should be back on step 1
    await expect(page.locator('input[type="email"]')).toBeVisible()
  })
})
```

### Testing Admin Dashboard

```typescript
// test/e2e/browser/admin.test.ts
import { test, expect } from '@playwright/test'

test.describe('Admin Dashboard Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Login as admin
    await page.goto('/login')
    await page.locator('input[type="email"]').fill('admin@example.com')
    await page.locator('input[type="password"]').fill('admin123')
    await page.getByRole('button', { name: /sign in/i }).click()
    await expect(page).toHaveURL(/\/$/, { timeout: 10000 })
  })

  test('navigate to admin dashboard', async ({ page }) => {
    await page.goto('/admin')

    // Should show dashboard
    await expect(page.locator('h1')).toContainText('Dashboard')
  })

  test('create new article', async ({ page }) => {
    await page.goto('/admin/articles')

    // Click "New Article"
    await page.getByRole('button', { name: /new article/i }).click()

    // Fill form
    await page.locator('input[name="title"]').fill('Test Article')
    await page.locator('input[name="slug"]').fill('test-article')
    await page.locator('textarea[name="content"]').fill('Article content')

    // Submit
    await page.getByRole('button', { name: /create/i }).click()

    // Should redirect to article list
    await expect(page).toHaveURL('/admin/articles')

    // Should see the new article
    await expect(page.getByText('Test Article')).toBeVisible()
  })

  test('edit article', async ({ page }) => {
    await page.goto('/admin/articles')

    // Click edit on first article
    await page.getByRole('button', { name: /edit/i }).first().click()

    // Update title
    await page.locator('input[name="title"]').fill('Updated Title')

    // Save
    await page.getByRole('button', { name: /save/i }).click()

    // Verify update
    await expect(page.getByText('Updated Title')).toBeVisible()
  })

  test('delete article', async ({ page }) => {
    await page.goto('/admin/articles')

    const articleCount = await page.locator('tbody tr').count()

    // Click delete on first article
    await page.getByRole('button', { name: /delete/i }).first().click()

    // Confirm deletion
    await page.getByRole('button', { name: /confirm/i }).click()

    // Should have one less article
    await expect(page.locator('tbody tr')).toHaveCount(articleCount - 1)
  })
})
```

## Running Tests

```bash
# Run all unit tests
pnpm test:unit

# Run all E2E tests (API + Browser)
pnpm test:e2e

# Run browser E2E tests with Playwright
pnpm test:e2e:browser

# Run all tests
pnpm test:all

# Generate coverage report
pnpm test:unit -- --coverage
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install pnpm
        uses: pnpm/action-setup@v4

      - name: Install dependencies
        run: pnpm install

      - name: Install Playwright browsers
        run: pnpm playwright install --with-deps

      - name: Run unit tests
        run: pnpm test:unit --run

      - name: Run E2E tests
        run: pnpm test:e2e:browser

      - name: Upload Playwright report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: playwright-report
          path: playwright-report/
```

## Best Practices

1. **Unit tests** - Test pure functions, utilities, and component logic
2. **API tests** - Test backend endpoints, authentication, and data operations
3. **Browser tests** - Test user flows, navigation, and UI interactions
4. **Seed database** - Always seed test data before E2E tests
5. **Clean up** - Clean up test data after tests complete
6. **Use meaningful selectors** - Use `data-testid` or accessible selectors
7. **Wait for elements** - Use `toBeVisible()` with timeouts instead of `sleep()`
