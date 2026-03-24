import { test, expect } from '@playwright/test'

test.describe('Article Scraper', () => {
  test('should display homepage', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('h1')).toContainText('Public Articles')
  })

  test('should navigate to login page', async ({ page }) => {
    await page.goto('/login')
    await expect(page.locator('h1')).toContainText('Login')
  })

  test('should navigate to register page', async ({ page }) => {
    await page.goto('/register')
    await expect(page.locator('h1')).toContainText('Register')
  })
})
