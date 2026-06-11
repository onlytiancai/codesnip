import { test, expect } from '@playwright/test'

test.describe('Auth Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Go to the home page first
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

    // Check form elements for step 1
    await expect(page.locator('input[type="email"]')).toBeVisible()
    await expect(page.getByPlaceholder(/create a password/i)).toBeVisible()

    // Check continue button
    await expect(page.getByRole('button', { name: /continue/i })).toBeVisible()

    // Check OAuth buttons
    await expect(page.getByRole('button', { name: /github/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /google/i })).toBeVisible()

    // Check link to login
    await expect(page.getByRole('link', { name: /sign in/i })).toBeVisible()
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
    await expect(page.getByText(/invalid email or password/i)).toBeVisible({ timeout: 5000 })
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

  test('homepage renders', async ({ page }) => {
    await page.goto('/')

    // Check that page loads
    await expect(page).toHaveURL('/')

    // Page should have some content
    const bodyText = await page.locator('body').textContent()
    expect(bodyText).toBeTruthy()
  })

  test('login form validation', async ({ page }) => {
    await page.goto('/login')

    // Try to submit without filling fields
    await page.getByRole('button', { name: /sign in/i }).click()

    // Form should show validation or prevent submission
    // Check that we're still on login page
    await expect(page).toHaveURL('/login')
  })

  test('register form step navigation', async ({ page }) => {
    await page.goto('/register')

    // Fill step 1
    await page.locator('input[type="email"]').fill('test@example.com')
    await page.getByPlaceholder(/create a password/i).fill('password123')
    await page.getByPlaceholder(/confirm your password/i).fill('password123')

    // Click continue
    await page.getByRole('button', { name: /continue/i }).click()

    // Should now be on step 2 (name field should appear)
    await expect(page.getByPlaceholder(/your full name/i)).toBeVisible()

    // Go back should work
    await page.getByRole('button', { name: /back/i }).click()

    // Should be back on step 1
    await expect(page.locator('input[type="email"]')).toBeVisible()
  })
})