# Authentication with nuxt-auth-utils

Complete guide to implementing authentication in Nuxt 4 using nuxt-auth-utils, including email/password login, registration, and OAuth (GitHub, Google).

## Version Compatibility

| Package | Version | Notes |
|---------|---------|-------|
| nuxt-auth-utils | ^0.5.29 | Use `defineOAuthGitHubEventHandler` and `defineOAuthGoogleEventHandler` |
| nuxt | ^4.3.1 | Nuxt 4 uses `app/` directory structure |
| @nuxt/ui | ^4.5.0 | Components use `title`/`description` props for accessibility |
| zod | ^4.3.6 | Schema validation for auth endpoints |
| bcryptjs | ^3.0.3 | Password hashing |
| prisma | ^6.19.2 | Use `@prisma/adapter-better-sqlite3` for SQLite |

## Installation

```bash
pnpm add nuxt-auth-utils
```

In `nuxt.config.ts`:

```typescript
export default defineNuxtConfig({
  modules: [
    "@nuxt/ui",
    "nuxt-auth-utils"
  ],
  // Optional: Configure auth utils
  auth: {
    // Configuration options if needed
  }
})
```

## Environment Variables

Add to `.env`:

```env
# Database
DATABASE_URL="file:./dev.db"

# OAuth - GitHub
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret

# OAuth - Google
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Session
NUXT_SESSION_PASSWORD=your_session_password_at_least_32_characters
```

Generate a session password:

```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

## Database Schema

Add OAuth models to `prisma/schema.prisma`:

```prisma
model User {
  id        Int      @id @default(autoincrement())
  email     String   @unique
  name      String?
  password  String?  // null for OAuth users
  avatar    String?
  role      String   @default("USER")  // 'USER' | 'ADMIN'
  createdAt DateTime @default(now())
  updatedAt DateTime @default(now())

  // Associated OAuth accounts
  accounts Account[]
  articles Article[]
}

model Account {
  id                Int    @id @default(autoincrement())
  userId            Int
  provider          String  // 'github' | 'google'
  providerAccountId String  // OAuth provider's user ID
  access_token      String?
  refresh_token     String?

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@unique([provider, providerAccountId])
}
```

Run migration:

```bash
pnpm prisma migrate dev
```

## Auth Utilities

Create `server/utils/db-auth.ts`:

```typescript
import { prisma } from "./db";

// Password hashing
async function dbHashPassword(password: string): Promise<string> {
  const bcrypt = await import("bcryptjs");
  return bcrypt.hash(password, 10);
}

async function dbVerifyPassword(password: string, hash: string): Promise<boolean> {
  const bcrypt = await import("bcryptjs");
  return bcrypt.compare(password, hash);
}

// User lookups
async function findUserByEmail(email: string) {
  return prisma.user.findUnique({
    where: { email },
    include: { accounts: true },
  });
}

async function findUserById(id: number) {
  return prisma.user.findUnique({
    where: { id },
    include: { accounts: true },
  });
}

// User creation
async function createUser(data: {
  email: string;
  name?: string;
  password?: string;
  avatar?: string
}) {
  return prisma.user.create({
    data,
    include: { accounts: true },
  });
}

// OAuth account linking
async function linkOAuthAccount(
  userId: number,
  provider: string,
  providerAccountId: string,
  accessToken?: string,
  refreshToken?: string
) {
  return prisma.account.create({
    data: {
      userId,
      provider,
      providerAccountId,
      access_token: accessToken,
      refresh_token: refreshToken,
    },
  });
}

async function findAccountByProvider(provider: string, providerAccountId: string) {
  return prisma.account.findUnique({
    where: {
      provider_providerAccountId: {
        provider,
        providerAccountId,
      },
    },
    include: { user: true },
  });
}

async function findUserByOAuth(provider: string, providerAccountId: string) {
  const account = await findAccountByProvider(provider, providerAccountId);
  return account?.user || null;
}

// Account management
async function unlinkOAuthAccount(userId: number, provider: string) {
  return prisma.account.deleteMany({
    where: {
      userId,
      provider,
    },
  });
}

async function updateUserPassword(userId: number, password: string) {
  const hashedPassword = await dbHashPassword(password);
  return prisma.user.update({
    where: { id: userId },
    data: { password: hashedPassword },
  });
}

async function removeUserPassword(userId: number) {
  return prisma.user.update({
    where: { id: userId },
    data: { password: null },
  });
}

async function isAdmin(userId: number): Promise<boolean> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { role: true },
  });
  return user?.role === 'ADMIN';
}

export {
  dbHashPassword,
  dbVerifyPassword,
  findUserByEmail,
  findUserById,
  createUser,
  linkOAuthAccount,
  findAccountByProvider,
  findUserByOAuth,
  unlinkOAuthAccount,
  updateUserPassword,
  removeUserPassword,
  isAdmin,
};
```

## Email/Password Authentication

### Register Endpoint

Create `server/api/auth/register.post.ts`:

```typescript
import { z } from 'zod';
import { dbHashPassword, findUserByEmail, createUser } from '../../utils/db-auth';

export default defineEventHandler(async (event) => {
  try {
    const body = await readBody(event);

    // Validate request body
    const schema = z.object({
      name: z.string().min(2, 'Name must be at least 2 characters'),
      email: z.string().email('Please enter a valid email address'),
      password: z.string().min(6, 'Password must be at least 6 characters'),
      confirmPassword: z.string(),
    }).refine((data) => data.password === data.confirmPassword, {
      message: 'Passwords do not match',
      path: ['confirmPassword'],
    });

    const parsed = schema.parse(body);

    // Check if email already exists
    const existingUser = await findUserByEmail(parsed.email);

    if (existingUser) {
      throw createError({
        statusCode: 409,
        message: 'Email already registered',
      });
    }

    // Hash password
    const hashedPassword = await dbHashPassword(parsed.password);

    // Create user
    const user = await createUser({
      email: parsed.email,
      name: parsed.name,
      password: hashedPassword,
    });

    // Set user session
    await setUserSession(event, {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        avatar: user.avatar,
        role: user.role,
        hasPassword: !!user.password,
      },
    });

    return {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        avatar: user.avatar,
        role: user.role,
      },
    };
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      const firstError = error.errors?.[0]?.message || 'Validation failed';
      throw createError({
        statusCode: 400,
        message: firstError,
      });
    }
    throw error;
  }
});
```

### Login Endpoint

Create `server/api/auth/login.post.ts`:

```typescript
import { z } from 'zod';
import { dbVerifyPassword, findUserByEmail } from '../../utils/db-auth';

export default defineEventHandler(async (event) => {
  try {
    const body = await readBody(event);

    // Validate request body
    const schema = z.object({
      email: z.string().email('Please enter a valid email address'),
      password: z.string().min(6, 'Password must be at least 6 characters'),
    });

    const parsed = schema.parse(body);

    // Find user
    const user = await findUserByEmail(parsed.email);

    if (!user || !user.password) {
      throw createError({
        statusCode: 401,
        message: 'Invalid email or password',
      });
    }

    // Verify password
    const isValid = await dbVerifyPassword(parsed.password, user.password);

    if (!isValid) {
      throw createError({
        statusCode: 401,
        message: 'Invalid email or password',
      });
    }

    // Set user session
    await setUserSession(event, {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        avatar: user.avatar,
        role: user.role,
        hasPassword: !!user.password,
      },
    });

    return {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        avatar: user.avatar,
        role: user.role,
      },
    };
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      const firstError = error.errors?.[0]?.message || 'Validation failed';
      throw createError({
        statusCode: 400,
        message: firstError,
      });
    }
    throw error;
  }
});
```

### Logout Endpoint

Create `server/api/auth/logout.post.ts`:

```typescript
export default defineEventHandler(async (event) => {
  await clearUserSession(event);
  return { success: true };
});
```

## OAuth Authentication

### GitHub OAuth

The recommended approach is to use `defineOAuthGitHubEventHandler` from nuxt-auth-utils:

```typescript
// server/routes/auth/github.get.ts
import { findUserByOAuth, createUser, linkOAuthAccount, findUserByEmail } from '../../utils/db-auth'

export default defineOAuthGitHubEventHandler({
  config: {
    emailRequired: true,
  },
  async onSuccess(event, { user }) {
    // First check if OAuth account already exists
    let dbUser = await findUserByOAuth('github', user.id.toString())

    if (dbUser) {
      // Already exists, login directly
      await setUserSession(event, {
        user: {
          id: dbUser.id,
          email: dbUser.email,
          name: dbUser.name,
          avatar: dbUser.avatar,
          role: dbUser.role,
          hasPassword: !!dbUser.password,
        },
      })
      return
    }

    // Check if user with same email exists
    dbUser = await findUserByEmail(user.email)

    if (dbUser) {
      // User exists, link GitHub account
      await linkOAuthAccount(
        dbUser.id,
        'github',
        user.id.toString(),
        user.accessToken,
        user.refreshToken
      )
      await setUserSession(event, {
        user: {
          id: dbUser.id,
          email: dbUser.email,
          name: dbUser.name,
          avatar: dbUser.avatar,
          role: dbUser.role,
          hasPassword: !!dbUser.password,
        },
      })
      return
    }

    // Create new user and link GitHub account
    const newUser = await createUser({
      email: user.email,
      name: user.name || user.login,
      avatar: user.avatar_url,
    })

    await linkOAuthAccount(
      newUser.id,
      'github',
      user.id.toString(),
      user.accessToken,
      user.refreshToken
    )

    await setUserSession(event, {
      user: {
        id: newUser.id,
        email: newUser.email,
        name: newUser.name,
        avatar: newUser.avatar,
        role: newUser.role,
        hasPassword: !!newUser.password,
      },
    })
  },
  onError(event, error) {
    console.error('GitHub OAuth error:', error)
    // Redirect to login with error message
    return sendRedirect(event, '/login?error=oauth_github_failed')
  },
})
```

### Google OAuth

```typescript
// server/routes/auth/google.get.ts
import { findUserByOAuth, createUser, linkOAuthAccount, findUserByEmail } from '../../utils/db-auth'

export default defineOAuthGoogleEventHandler({
  config: {
    scope: ['openid', 'email', 'profile'],
  },
  async onSuccess(event, { user }) {
    // First check if OAuth account already exists
    let dbUser = await findUserByOAuth('google', user.sub)

    if (dbUser) {
      // Already exists, login directly
      await setUserSession(event, {
        user: {
          id: dbUser.id,
          email: dbUser.email,
          name: dbUser.name,
          avatar: dbUser.picture,
          role: dbUser.role,
          hasPassword: !!dbUser.password,
        },
      })
      return
    }

    // Check if user with same email exists
    dbUser = await findUserByEmail(user.email)

    if (dbUser) {
      // User exists, link Google account
      await linkOAuthAccount(
        dbUser.id,
        'google',
        user.sub,
        user.accessToken,
        user.refreshToken
      )
      await setUserSession(event, {
        user: {
          id: dbUser.id,
          email: dbUser.email,
          name: dbUser.name,
          avatar: dbUser.picture,
          role: dbUser.role,
          hasPassword: !!dbUser.password,
        },
      })
      return
    }

    // Create new user and link Google account
    const newUser = await createUser({
      email: user.email,
      name: user.name,
      avatar: user.picture,
    })

    await linkOAuthAccount(
      newUser.id,
      'google',
      user.sub,
      user.accessToken,
      user.refreshToken
    )

    await setUserSession(event, {
      user: {
        id: newUser.id,
        email: newUser.email,
        name: newUser.name,
        avatar: newUser.avatar,
        role: newUser.role,
        hasPassword: !!newUser.password,
      },
    })
  },
  onError(event, error) {
    console.error('Google OAuth error:', error)
    // Redirect to login with error message
    return sendRedirect(event, '/login?error=oauth_google_failed')
  },
})
```

### OAuth Error Handling

Handle common OAuth failure scenarios gracefully:

#### 1. User Denies Permission / Cancels Flow

When a user cancels the OAuth flow or denies permission, the OAuth provider returns an error:

```typescript
// server/routes/auth/github.get.ts
export default defineOAuthGitHubEventHandler({
  config: {
    emailRequired: true,
  },
  async onSuccess(event, { user }) {
    // ... success handling
  },
  async onError(event, error) {
    console.error('GitHub OAuth error:', error)

    // Check for specific error types
    if (error.message?.includes('access_denied')) {
      // User denied permission or cancelled
      return sendRedirect(event, '/login?error=oauth_cancelled')
    }

    if (error.message?.includes('invalid_code')) {
      // Expired or invalid authorization code
      return sendRedirect(event, '/login?error=oauth_expired')
    }

    // Generic error
    return sendRedirect(event, '/login?error=oauth_failed')
  },
})
```

#### 2. Missing Email from Provider

Some users may have private email addresses on GitHub. Handle this case:

```typescript
async onSuccess(event, { user }) {
  // GitHub may return null email if user has no public email
  if (!user.email) {
    try {
      // Fetch emails from GitHub API
      const emails: any[] = await $fetch('https://api.github.com/user/emails', {
        headers: {
          Authorization: `token ${user.accessToken}`,
        },
      })

      const primaryEmail = emails.find((e: any) => e.primary)
      user.email = primaryEmail?.email
    } catch (fetchError) {
      console.error('Failed to fetch GitHub emails:', fetchError)
      return sendRedirect(event, '/login?error=no_email')
    }
  }

  if (!user.email) {
    // Still no email - can't create user without one
    return sendRedirect(event, '/login?error=no_email')
  }

  // Continue with user creation/login...
}
```

#### 3. Account Already Linked to Different Provider

When a user tries to link GitHub but their email is already registered with Google:

```typescript
async onSuccess(event, { user }) {
  // Check if OAuth account already exists
  let dbUser = await findUserByOAuth('github', user.id.toString())

  if (dbUser) {
    // Already linked, just log them in
    await setUserSession(event, { user: { ... } })
    return
  }

  // Check if email exists
  const existingUser = await findUserByEmail(user.email)

  if (existingUser) {
    // Email exists - check if they're trying to add a second OAuth provider
    const hasGitHub = existingUser.accounts.some((a: any) => a.provider === 'github')

    if (hasGitHub) {
      // Already has GitHub linked - should use existing login
      return sendRedirect(event, '/login?error=account_exists_use_password')
    }

    // Link GitHub to existing account (e.g., user registered with Google, now adding GitHub)
    await linkOAuthAccount(existingUser.id, 'github', user.id.toString(), user.accessToken)
    await setUserSession(event, { user: { ... } })
    return
  }

  // Create new user...
}
```

#### 4. Frontend Error Display

Show user-friendly error messages on the login page:

```vue
<!-- app/pages/login.vue -->
<script setup lang="ts">
const route = useRoute()

const errorMessages: Record<string, string> = {
  oauth_cancelled: 'You cancelled the sign-in process',
  oauth_expired: 'The authorization code expired. Please try again',
  oauth_failed: 'Failed to sign in with OAuth provider',
  no_email: 'Could not get your email from the OAuth provider',
  account_exists_use_password: 'This email is already registered. Please sign in with your password',
}

const oauthError = computed(() => {
  const error = route.query.error as string
  return errorMessages[error] || null
})
</script>

<template>
  <div>
    <div v-if="oauthError" class="mb-4 p-3 bg-red-50 text-red-600 rounded">
      {{ oauthError }}
    </div>
    <!-- Rest of login form -->
  </div>
</template>
```

## Frontend Components

### Login Page

```vue
<!-- app/pages/login.vue -->
<script setup lang="ts">
const email = ref('');
const password = ref('');
const error = ref('');
const loading = ref(false);

async function handleLogin() {
  error.value = '';
  loading.value = true;

  try {
    await $fetch('/api/auth/login', {
      method: 'POST',
      body: {
        email: email.value,
        password: password.value,
      },
    });

    navigateTo('/');
  } catch (e: any) {
    error.value = e.data?.message || 'Login failed';
  } finally {
    loading.value = false;
  }
}

const githubOAuthUrl = `https://github.com/login/oauth/authorize?client_id=${process.env.NUXT_PUBLIC_GITHUB_CLIENT_ID}&scope=user:email`;
const googleOAuthUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${process.env.NUXT_PUBLIC_GOOGLE_CLIENT_ID}&redirect_uri=${process.env.NUXT_PUBLIC_BASE_URL}/auth/google&response_type=code&scope=openid%20profile%20email`;
</script>

<template>
  <div class="min-h-screen flex items-center justify-center">
    <UCard class="w-full max-w-md">
      <h1 class="text-2xl font-bold text-center mb-6">Welcome Back</h1>

      <form @submit.prevent="handleLogin" class="space-y-4">
        <UInput
          v-model="email"
          type="email"
          placeholder="Email"
          required
        />

        <UInput
          v-model="password"
          type="password"
          placeholder="Password"
          required
        />

        <UButton
          type="submit"
          color="primary"
          class="w-full"
          :loading="loading"
        >
          Sign In
        </UButton>

        <div v-if="error" class="text-red-500 text-sm text-center">
          {{ error }}
        </div>
      </form>

      <div class="my-6 flex items-center">
        <div class="flex-1 border-t border-gray-300" />
        <span class="px-4 text-sm text-gray-500">Or continue with</span>
        <div class="flex-1 border-t border-gray-300" />
      </div>

      <div class="space-y-3">
        <UButton
          :to="githubOAuthUrl"
          color="neutral"
          variant="outline"
          class="w-full"
          icon="lucide-github"
        >
          Continue with GitHub
        </UButton>

        <UButton
          :to="googleOAuthUrl"
          color="neutral"
          variant="outline"
          class="w-full"
          icon="lucide-mail"
        >
          Continue with Google
        </UButton>
      </div>

      <p class="mt-6 text-center text-sm">
        Don't have an account?
        <NuxtLink to="/register" class="text-primary hover:underline">
          Sign up
        </NuxtLink>
      </p>
    </UCard>
  </div>
</template>
```

### Session Management

```vue
<!-- In any component -->
<script setup lang="ts">
const { data: session, refresh } = useUserSession();

async function logout() {
  await $fetch('/api/auth/logout', { method: 'POST' });
  await refresh();
  navigateTo('/login');
}
</script>

<template>
  <div v-if="session">
    <p>Welcome, {{ session.user?.name }}</p>
    <UButton @click="logout">Logout</UButton>
  </div>
  <NuxtLink v-else to="/login">Login</NuxtLink>
</template>
```

### Protected Routes

Create `server/middleware/admin.ts`:

```typescript
export default defineEventHandler(async (event) => {
  // Only protect admin routes
  if (!event.path.startsWith('/api/admin')) {
    return;
  }

  const session = await getUserSession(event);

  if (!session || !session.user) {
    throw createError({
      statusCode: 401,
      message: 'Unauthorized',
    });
  }

  // Check admin role
  if (session.user.role !== 'ADMIN') {
    throw createError({
      statusCode: 403,
      message: 'Admin access required',
    });
  }
});
```

## OAuth Setup

### GitHub OAuth App

1. Go to GitHub Settings → Developer Settings → OAuth Apps
2. Click "New OAuth App"
3. Fill in:
   - **Application name**: Your App Name
   - **Homepage URL**: `http://localhost:3000` (dev) or your production URL
   - **Authorization callback URL**: `http://localhost:3000/auth/github`
4. Copy Client ID and generate Client Secret
5. Add to `.env`

### Google OAuth Credentials

1. Go to Google Cloud Console → APIs & Services → Credentials
2. Click "Create Credentials" → "OAuth Client ID"
3. Application type: **Web application**
4. Authorized JavaScript origins: `http://localhost:3000`
5. Authorized redirect URIs: `http://localhost:3000/auth/google`
6. Copy Client ID and Client Secret
7. Add to `.env`

## Tips

1. **Always use HTTPS** in production for OAuth callbacks
2. **Store session password** in environment variables (min 32 characters)
3. **Use Zod validation** for all auth endpoints
4. **Handle OAuth errors** gracefully with user-friendly messages
5. **Link OAuth accounts** to existing email users automatically
6. **Clear session on logout** using `clearUserSession()`

## Protected Routes and Role-Based UI

### Client-Side Middleware for Admin Routes

Create `app/middleware/admin.ts` to protect admin pages:

```typescript
// app/middleware/admin.ts
export default defineNuxtRouteMiddleware(async (to) => {
  // Only protect /admin routes
  if (!to.path.startsWith('/admin')) {
    return
  }

  const { user } = await useUserSession()

  // Not logged in - redirect to login
  if (!user.value) {
    return navigateTo('/login', {
      query: { redirect: to.fullPath }
    })
  }

  // Not admin - redirect to home
  if (user.value.role !== 'ADMIN') {
    return navigateTo('/')
  }
})
```

Apply the middleware in admin pages:

```vue
<!-- app/pages/admin/users/index.vue -->
<script setup lang="ts">
definePageMeta({
  layout: 'admin',
  middleware: 'admin'
})
</script>

<template>
  <NuxtLayout name="admin">
    <!-- Admin content -->
  </NuxtLayout>
</template>
```

### Role-Based Component Rendering

Show/hide UI elements based on user role:

```vue
<script setup lang="ts">
const { user } = useUserSession()

const isAdmin = computed(() => user.value?.role === 'ADMIN')
</script>

<template>
  <div>
    <!-- Visible to all users -->
    <UButton icon="lucide-home">Home</UButton>

    <!-- Only visible to admins -->
    <UButton
      v-if="isAdmin"
      icon="lucide-shield"
      color="warning"
      to="/admin"
    >
      Admin Dashboard
    </UButton>

    <!-- Admin-only actions in tables -->
    <UTable :data="users" :columns="columns">
      <template #actions-cell="{ row }">
        <UDropdownMenu :items="getActionItems(row.original)">
          <UButton
            icon="lucide-more-horizontal"
            color="neutral"
            variant="ghost"
            size="xs"
          />
        </UDropdownMenu>
      </template>
    </UTable>
  </div>
</template>
```

### Admin Layout with Navigation

Create a dedicated admin layout:

```vue
<!-- app/layouts/admin.vue -->
<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Admin Header -->
    <header class="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
      <div class="flex items-center justify-between px-6 py-4">
        <div class="flex items-center gap-4">
          <NuxtLink to="/admin" class="text-lg font-bold">
            Admin Dashboard
          </NuxtLink>

          <!-- Admin Navigation -->
          <nav class="flex items-center gap-2 ml-8">
            <NuxtLink
              to="/admin/users"
              class="px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              Users
            </NuxtLink>
            <NuxtLink
              to="/admin/articles"
              class="px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              Articles
            </NuxtLink>
            <NuxtLink
              to="/admin/categories"
              class="px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              Categories
            </NuxtLink>
            <NuxtLink
              to="/admin/analytics"
              class="px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              Analytics
            </NuxtLink>
          </nav>
        </div>

        <div class="flex items-center gap-4">
          <NuxtLink to="/" class="text-sm text-gray-500">
            Back to Site
          </NuxtLink>
          <UButton icon="lucide-log-out" variant="outline" @click="logout">
            Logout
          </UButton>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="p-6">
      <slot />
    </main>
  </div>
</template>

<script setup lang="ts">
async function logout() {
  await $fetch('/api/auth/logout', { method: 'POST' })
  navigateTo('/')
}
</script>
```

### User Stats Dashboard Example

Display role-based statistics in an admin dashboard:

```vue
<script setup lang="ts">
const { users, loading } = useAdminUsers()

const adminCount = computed(() => users.value.filter(u => u.role === 'ADMIN').length)
const userCount = computed(() => users.value.filter(u => u.role === 'USER').length)
const totalArticles = computed(() => users.value.reduce((sum, u) => sum + u.articleCount, 0))
</script>

<template>
  <div class="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-6">
    <UCard class="text-center">
      <p class="text-2xl font-bold">{{ pagination.total }}</p>
      <p class="text-sm text-gray-500 dark:text-gray-400">Total Users</p>
    </UCard>
    <UCard class="text-center">
      <p class="text-2xl font-bold text-purple-500">{{ adminCount }}</p>
      <p class="text-sm text-gray-500 dark:text-gray-400">Admins</p>
    </UCard>
    <UCard class="text-center">
      <p class="text-2xl font-bold text-blue-500">{{ userCount }}</p>
      <p class="text-sm text-gray-500 dark:text-gray-400">Regular Users</p>
    </UCard>
    <UCard class="text-center">
      <p class="text-2xl font-bold text-green-500">{{ totalArticles }}</p>
      <p class="text-sm text-gray-500 dark:text-gray-400">Total Articles</p>
    </UCard>
  </div>
</template>
```

### Action Menus with Role-Based Options

```vue
<script setup lang="ts">
const { user } = useUserSession()

const getActionItems = (user: any) => {
  const items = [
    [{
      label: 'View Profile',
      icon: 'lucide-user',
      click: () => viewProfile(user)
    }]
  ]

  // Admin-only actions
  if (user.value?.role === 'ADMIN') {
    items.push([{
      label: 'Edit User',
      icon: 'lucide-edit',
      click: () => editUser(user)
    }, {
      label: 'Delete User',
      icon: 'lucide-trash-2',
      color: 'error' as const,
      click: () => deleteUser(user)
    }])
  }

  return items
}
</script>

<template>
  <UTable :data="users" :columns="columns">
    <template #actions-cell="{ row }">
      <UDropdownMenu :items="getActionItems(row.original)">
        <UButton icon="lucide-more-horizontal" color="neutral" variant="ghost" size="xs" />
      </UDropdownMenu>
    </template>
  </UTable>
</template>
```
