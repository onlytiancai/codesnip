# Authentication with nuxt-auth-utils

Complete guide to implementing authentication in Nuxt 4 using nuxt-auth-utils, including email/password login, registration, and OAuth (GitHub, Google).

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

Create `server/routes/auth/github.get.ts`:

```typescript
import { z } from 'zod';

export default defineEventHandler(async (event) => {
  const query = await getQuery(z.object({ code: z.string() }).parse(await getQuery(event)));

  // Exchange code for access token
  const tokenResponse = await $fetch('https://github.com/login/oauth/access_token', {
    method: 'POST',
    body: {
      client_id: process.env.GITHUB_CLIENT_ID,
      client_secret: process.env.GITHUB_CLIENT_SECRET,
      code: query.code,
    },
  });

  const token = new URLSearchParams(tokenResponse as string).get('access_token');

  if (!token) {
    throw createError({
      statusCode: 400,
      message: 'Failed to get access token from GitHub',
    });
  }

  // Get user profile
  const userProfile: any = await $fetch('https://api.github.com/user', {
    headers: {
      Authorization: `token ${token}`,
    },
  });

  // Get user email
  let email = userProfile.email;

  if (!email) {
    const emails: any[] = await $fetch('https://api.github.com/user/emails', {
      headers: {
        Authorization: `token ${token}`,
      },
    });

    const primaryEmail = emails.find((e: any) => e.primary);
    email = primaryEmail?.email;
  }

  if (!email) {
    throw createError({
      statusCode: 400,
      message: 'No email found in GitHub profile',
    });
  }

  // Check if user exists or create new
  const { linkOAuthAccount, findUserByOAuth, createUser } = await import('../../utils/db-auth');

  let user = await findUserByOAuth('github', userProfile.id.toString());

  if (!user) {
    // Check if email already exists
    const { findUserByEmail } = await import('../../utils/db-auth');
    user = await findUserByEmail(email);

    if (user) {
      // Link GitHub account to existing user
      await linkOAuthAccount(
        user.id,
        'github',
        userProfile.id.toString(),
        token
      );
    } else {
      // Create new user
      user = await createUser({
        email,
        name: userProfile.name || userProfile.login,
        avatar: userProfile.avatar_url,
      });

      // Link GitHub account
      await linkOAuthAccount(
        user.id,
        'github',
        userProfile.id.toString(),
        token
      );
    }
  }

  // Set session
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

  return sendRedirect(event, '/');
});
```

### Google OAuth

Create `server/routes/auth/google.get.ts`:

```typescript
import { z } from 'zod';

export default defineEventHandler(async (event) => {
  const query = await getQuery(z.object({ code: z.string() }).parse(await getQuery(event)));

  // Exchange code for tokens
  const tokenResponse: any = await $fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    body: {
      client_id: process.env.GOOGLE_CLIENT_ID,
      client_secret: process.env.GOOGLE_CLIENT_SECRET,
      code: query.code,
      grant_type: 'authorization_code',
      redirect_uri: `${process.env.NUXT_PUBLIC_BASE_URL}/auth/google`,
    },
  });

  const { access_token } = tokenResponse;

  // Get user profile
  const userProfile: any = await $fetch('https://www.googleapis.com/oauth2/v2/userinfo', {
    headers: {
      Authorization: `Bearer ${access_token}`,
    },
  });

  // Check if user exists or create new
  const { linkOAuthAccount, findUserByOAuth, createUser } = await import('../../utils/db-auth');

  let user = await findUserByOAuth('google', userProfile.id);

  if (!user) {
    const { findUserByEmail } = await import('../../utils/db-auth');
    user = await findUserByEmail(userProfile.email);

    if (user) {
      await linkOAuthAccount(
        user.id,
        'google',
        userProfile.id,
        access_token
      );
    } else {
      user = await createUser({
        email: userProfile.email,
        name: userProfile.name,
        avatar: userProfile.picture,
      });

      await linkOAuthAccount(
        user.id,
        'google',
        userProfile.id,
        access_token
      );
    }
  }

  // Set session
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

  return sendRedirect(event, '/');
});
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
