import { test, expect } from '@playwright/test'

test.describe('User Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Start from the home page
    await page.goto('/')
  })

  test('should show login link when not authenticated', async ({ page }) => {
    // Check that Sign In button is visible
    await expect(page.getByRole('link', { name: 'Sign In' })).toBeVisible()
  })

  test('should login successfully with valid credentials', async ({ page }) => {
    // Go to login page
    await page.goto('/login')

    // Fill in credentials
    await page.getByLabel('Email').fill('user@example.com')
    await page.getByLabel('Password').fill('user123')

    // Submit form
    await page.getByRole('button', { name: 'Sign In' }).click()

    // Wait for redirect to home
    await page.waitForURL('/')

    // Check that user menu is visible
    await expect(page.getByRole('button', { name: 'Test User' })).toBeVisible()
  })

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login')

    await page.getByLabel('Email').fill('user@example.com')
    await page.getByLabel('Password').fill('wrongpassword')

    await page.getByRole('button', { name: 'Sign In' }).click()

    // Check for error toast
    await expect(page.getByText('Login Failed')).toBeVisible()
  })

  test('should logout successfully', async ({ page }) => {
    // Login first
    await page.goto('/login')
    await page.getByLabel('Email').fill('user@example.com')
    await page.getByLabel('Password').fill('user123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await page.waitForURL('/')

    // Click user menu
    await page.getByRole('button', { name: 'Test User' }).click()

    // Click logout
    await page.getByRole('menuitem', { name: 'Logout' }).click()

    // Wait for redirect
    await page.waitForURL('/')

    // Check that Sign In button is visible again
    await expect(page.getByRole('link', { name: 'Sign In' })).toBeVisible()
  })
})

test.describe('User Profile Page', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await page.goto('/login')
    await page.getByLabel('Email').fill('user@example.com')
    await page.getByLabel('Password').fill('user123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await page.waitForURL('/')
  })

  test('should display user profile', async ({ page }) => {
    await page.goto('/user')

    // Check profile elements
    await expect(page.getByRole('heading', { name: 'Test User' })).toBeVisible()
    await expect(page.getByText('user@example.com')).toBeVisible()
  })

  test('should display stats cards', async ({ page }) => {
    await page.goto('/user')

    // Check stats are displayed
    await expect(page.getByText('Articles Read')).toBeVisible()
    await expect(page.getByText('Reading Time')).toBeVisible()
    await expect(page.getByText('Words Learned')).toBeVisible()
    await expect(page.getByText('Day Streak')).toBeVisible()
  })

  test('should navigate to quick actions', async ({ page }) => {
    await page.goto('/user')

    // Click Reading History
    await page.getByRole('link', { name: 'Reading History' }).first().click()
    await expect(page).toHaveURL('/user/history')
  })
})

test.describe('Reading History Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login')
    await page.getByLabel('Email').fill('user@example.com')
    await page.getByLabel('Password').fill('user123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await page.waitForURL('/')
  })

  test('should display reading history page', async ({ page }) => {
    await page.goto('/user/history')

    await expect(page.getByRole('heading', { name: 'Reading History' })).toBeVisible()
    await expect(page.getByText('Articles Read')).toBeVisible()
    await expect(page.getByText('Total Time')).toBeVisible()
    await expect(page.getByText('Day Streak')).toBeVisible()
  })
})

test.describe('Bookmarks Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login')
    await page.getByLabel('Email').fill('user@example.com')
    await page.getByLabel('Password').fill('user123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await page.waitForURL('/')
  })

  test('should display bookmarks page', async ({ page }) => {
    await page.goto('/user/bookmarks')

    await expect(page.getByRole('heading', { name: 'Bookmarks' })).toBeVisible()
  })
})

test.describe('Vocabulary Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login')
    await page.getByLabel('Email').fill('user@example.com')
    await page.getByLabel('Password').fill('user123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await page.waitForURL('/')
  })

  test('should display vocabulary page', async ({ page }) => {
    await page.goto('/user/vocabulary')

    await expect(page.getByRole('heading', { name: 'Vocabulary' })).toBeVisible()
    await expect(page.getByText('Total Words')).toBeVisible()
    await expect(page.getByText('Mastered')).toBeVisible()
    await expect(page.getByText('Learning')).toBeVisible()
  })

  test('should open add word modal', async ({ page }) => {
    await page.goto('/user/vocabulary')

    // Click Add Word button
    await page.getByRole('button', { name: 'Add Word' }).click()

    // Check modal is open
    await expect(page.getByRole('heading', { name: 'Add New Word' })).toBeVisible()
  })
})

test.describe('Settings Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login')
    await page.getByLabel('Email').fill('user@example.com')
    await page.getByLabel('Password').fill('user123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await page.waitForURL('/')
  })

  test('should display settings page', async ({ page }) => {
    await page.goto('/user/settings')

    await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Profile' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Preferences' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Appearance' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Notifications' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Account' })).toBeVisible()
  })

  test('should switch between tabs', async ({ page }) => {
    await page.goto('/user/settings')

    // Click Preferences tab
    await page.getByRole('tab', { name: 'Preferences' }).click()
    await expect(page.getByRole('heading', { name: 'Reading Preferences' })).toBeVisible()

    // Click Appearance tab
    await page.getByRole('tab', { name: 'Appearance' }).click()
    await expect(page.getByRole('heading', { name: 'Appearance' })).toBeVisible()
  })

  test('should update profile', async ({ page }) => {
    await page.goto('/user/settings')

    // Update name
    await page.getByLabel('Full Name').fill('Updated Name')
    await page.getByRole('button', { name: 'Save Changes' }).click()

    // Check success toast
    await expect(page.getByText('Profile updated')).toBeVisible()
  })
})

test.describe('Protected Routes', () => {
  test('should redirect to login when not authenticated', async ({ page }) => {
    await page.goto('/user')

    // Should be redirected to login
    await expect(page).toHaveURL('/login')
    await expect(page.getByRole('heading', { name: 'Welcome Back' })).toBeVisible()
  })

  test('should redirect to intended page after login', async ({ page }) => {
    // Try to access protected page
    await page.goto('/user/vocabulary')

    // Should be on login page
    await expect(page).toHaveURL(/\/login/)

    // Login
    await page.getByLabel('Email').fill('user@example.com')
    await page.getByLabel('Password').fill('user123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    // Should be redirected to vocabulary page
    await expect(page).toHaveURL('/user/vocabulary')
  })
})

test.describe('Admin Access', () => {
  test('admin should see admin dashboard link', async ({ page }) => {
    // Login as admin
    await page.goto('/login')
    await page.getByLabel('Email').fill('admin@example.com')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await page.waitForURL('/')

    // Click user menu
    await page.getByRole('button', { name: 'Admin User' }).click()

    // Check Admin Dashboard link is visible
    await expect(page.getByRole('menuitem', { name: 'Admin Dashboard' })).toBeVisible()
  })

  test('regular user should not see admin dashboard link', async ({ page }) => {
    // Login as regular user
    await page.goto('/login')
    await page.getByLabel('Email').fill('user@example.com')
    await page.getByLabel('Password').fill('user123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await page.waitForURL('/')

    // Click user menu
    await page.getByRole('button', { name: 'Test User' }).click()

    // Check Admin Dashboard link is not visible
    await expect(page.getByRole('menuitem', { name: 'Admin Dashboard' })).not.toBeVisible()
  })
})