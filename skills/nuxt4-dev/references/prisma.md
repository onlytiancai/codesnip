# Prisma ORM in Nuxt 4

Complete guide to using Prisma ORM in Nuxt 4 applications with SQLite or PostgreSQL.

## Initial Setup

### 1. Install Dependencies

```bash
pnpm add prisma @prisma/client
pnpm add -D @types/better-sqlite3  # For SQLite
pnpm add @prisma/adapter-better-sqlite3 better-sqlite3  # SQLite only
```

### 2. Initialize Prisma

```bash
pnpm prisma init
```

This creates:
- `prisma/schema.prisma` - Database schema file
- `.env` - Environment variables (includes `DATABASE_URL`)

### 3. Configure Database

**For SQLite** (development/local):

```env
DATABASE_URL="file:./dev.db"
```

```prisma
datasource db {
  provider = "sqlite"
  url      = env("DATABASE_URL")
}
```

**For PostgreSQL** (production):

```env
DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
```

```prisma
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}
```

## Schema Design

### Basic Model

```prisma
model User {
  id        Int      @id @default(autoincrement())
  email     String   @unique
  name      String?
  password  String?
  avatar    String?
  role      String   @default("USER")  // 'USER' | 'ADMIN'
  createdAt DateTime @default(now())
  updatedAt DateTime @default(now())

  articles Article[]
}
```

### Relations

```prisma
model Article {
  id         Int      @id @default(autoincrement())
  title      String
  slug       String   @unique
  content    String?
  status     String   @default("draft")
  categoryId Int?
  category   Category? @relation(fields: [categoryId], references: [id])
  tags       ArticleTag[]
  authorId   Int
  author     User     @relation(fields: [authorId], references: [id])
}

model Category {
  id       Int       @id @default(autoincrement())
  name     String    @unique
  slug     String    @unique
  articles Article[]
}

model Tag {
  id       Int          @id @default(autoincrement())
  name     String       @unique
  slug     String       @unique
  articles ArticleTag[]
}

// Many-to-many join table
model ArticleTag {
  articleId Int
  tagId     Int
  article   Article @relation(fields: [articleId], references: [id], onDelete: Cascade)
  tag       Tag     @relation(fields: [tagId], references: [id], onDelete: Cascade)
  @@id([articleId, tagId])
}
```

### OAuth Models

```prisma
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

## Migration Workflow

### After Schema Changes

**Every time you modify `schema.prisma`:**

```bash
# Development - creates migration and applies it
pnpm prisma migrate dev

# Production - applies existing migrations only
pnpm prisma migrate deploy
```

### Generate Prisma Client

After migrations or when deploying:

```bash
pnpm prisma generate
```

This generates the Prisma Client in your configured output directory.

**Note:** In Nuxt 4, configure the output to be inside your project:

```prisma
generator client {
  provider = "prisma-client"
  output   = "../generated/prisma"
}
```

## Prisma Client Setup for Nuxt 4

Create `server/utils/db.ts`:

```typescript
import "dotenv/config";
import { PrismaBetterSQLite3 } from "@prisma/adapter-better-sqlite3";
import { PrismaClient } from "../../generated/prisma/client";

const connectionString = `${process.env.DATABASE_URL}`;

const prismaClientSingleton = () => {
  const adapter = new PrismaBetterSQLite3({ url: connectionString });
  return new PrismaClient({ adapter });
};

type PrismaClientSingleton = ReturnType<typeof prismaClientSingleton>;

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClientSingleton | undefined;
};

export const prisma = globalForPrisma.prisma ?? prismaClientSingleton();

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
```

**Why this pattern?**
- Nuxt 4 auto-imports exports from `server/utils/`
- Singleton pattern prevents multiple Prisma instances in development
- Adapter pattern supports different databases

## Usage in API Routes

### Basic CRUD

```typescript
// server/api/admin/articles/index.get.ts
export default defineEventHandler(async (event) => {
  const articles = await prisma.article.findMany({
    include: {
      category: true,
      tags: { include: { tag: true } },
      author: { select: { id: true, name: true, email: true } }
    },
    orderBy: { createdAt: 'desc' }
  });

  return articles;
});
```

### Create with Validation

```typescript
// server/api/admin/articles/index.post.ts
import { z } from 'zod';

export default defineEventHandler(async (event) => {
  const body = await readBody(event);

  const schema = z.object({
    title: z.string().min(1),
    slug: z.string().min(1),
    content: z.string().optional(),
    categoryId: z.number().optional(),
    tagIds: z.array(z.number()).optional(),
  });

  const parsed = schema.parse(body);

  // Create article
  const article = await prisma.article.create({
    data: {
      ...parsed,
      authorId: 1, // Get from session
    },
  });

  // Create tag relations if provided
  if (parsed.tagIds?.length) {
    await prisma.articleTag.createMany({
      data: parsed.tagIds.map(tagId => ({
        articleId: article.id,
        tagId,
      })),
    });
  }

  return article;
});
```

### Update with Relations

```typescript
// server/api/admin/articles/[id].put.ts
export default defineEventHandler(async (event) => {
  const id = Number(event.context.params?.id);
  const body = await readBody(event);

  // Delete existing tag relations
  await prisma.articleTag.deleteMany({
    where: { articleId: id },
  });

  // Update article and create new tag relations
  const article = await prisma.article.update({
    where: { id },
    data: {
      ...body,
      tags: body.tagIds?.length ? {
        create: body.tagIds.map((tagId: number) => ({
          tag: { connect: { id: tagId } },
        })),
      } : undefined,
    },
    include: {
      tags: { include: { tag: true } },
    },
  });

  return article;
});
```

### Delete

```typescript
// server/api/admin/tags/[id].delete.ts
export default defineEventHandler(async (event) => {
  const id = Number(event.context.params?.id);

  await prisma.tag.delete({
    where: { id },
  });

  return { success: true };
});
```

## Database Auth Utilities

Create `server/utils/db-auth.ts`:

```typescript
import { prisma } from "./db";

async function dbHashPassword(password: string): Promise<string> {
  const bcrypt = await import("bcryptjs");
  return bcrypt.hash(password, 10);
}

async function dbVerifyPassword(password: string, hash: string): Promise<boolean> {
  const bcrypt = await import("bcryptjs");
  return bcrypt.compare(password, hash);
}

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

// Export all functions
export {
  dbHashPassword,
  dbVerifyPassword,
  findUserByEmail,
  findUserById,
  createUser,
};
```

## Seed Database

Create `prisma/seed.ts`:

```typescript
import { PrismaClient } from '../generated/prisma/client'
import bcrypt from 'bcryptjs'

const prisma = new PrismaClient()

async function main() {
  // Create admin user
  const hashedPassword = await bcrypt.hash('admin123', 10)

  await prisma.user.upsert({
    where: { email: 'admin@example.com' },
    update: {},
    create: {
      email: 'admin@example.com',
      name: 'Admin User',
      password: hashedPassword,
      role: 'ADMIN',
    },
  })

  // Create categories
  await prisma.category.createMany({
    data: [
      { name: 'Tutorial', slug: 'tutorial' },
      { name: 'Guide', slug: 'guide' },
      { name: 'Reference', slug: 'reference' },
    ],
    skipDuplicates: true,
  })
}

main()
  .then(() => prisma.$disconnect())
  .catch(async (e) => {
    console.error(e)
    await prisma.$disconnect()
    process.exit(1)
  })
```

Add to `package.json`:

```json
{
  "scripts": {
    "db:seed": "tsx prisma/seed.ts"
  }
}
```

Run seeding:

```bash
pnpm db:seed
```

## Common Queries

### Find with Relations

```typescript
// Find article with all relations
const article = await prisma.article.findUnique({
  where: { id },
  include: {
    category: true,
    tags: { include: { tag: true } },
    author: { select: { id: true, name: true, email: true } },
    sentences: { orderBy: { order: 'asc' } }
  }
});
```

### Filter and Search

```typescript
// Search articles
const articles = await prisma.article.findMany({
  where: {
    status: 'published',
    OR: [
      { title: { contains: searchQuery } },
      { excerpt: { contains: searchQuery } }
    ]
  },
  orderBy: { publishAt: 'desc' },
  take: 10,
  skip: (page - 1) * 10
});
```

### Aggregations

```typescript
// Get analytics
const overview = await prisma.article.aggregate({
  _count: true,
  _sum: { views: true },
  _avg: { views: true }
});

// Group by category
const byCategory = await prisma.article.groupBy({
  by: ['categoryId'],
  _count: true
});
```

## Tips

1. **Always run `prisma generate`** after schema changes
2. **Use transactions** for multiple related operations:
   ```typescript
   await prisma.$transaction(async (tx) => {
     await tx.article.create({...});
     await tx.articleTag.createMany({...});
   });
   ```
3. **Include relations explicitly** - Prisma doesn't auto-include
4. **Use `findUnique`** for single lookups by ID or unique field
5. **Use `findFirst`** when you need filtering with relations
