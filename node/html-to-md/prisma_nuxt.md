# Build a Nuxt app with Prisma ORM and Prisma Postgres

Frameworks

A step-by-step guide to setting up and using Prisma ORM and Prisma Postgres in a Nuxt app

This guide shows you how to set up Prisma ORM in a Nuxt application with [Prisma Postgres](https://prisma.io/postgres).

-   Node.js 18+
-   A [Prisma Postgres](https://console.prisma.io/) database (or any PostgreSQL database)

Create a new Nuxt project:

Navigate to the project and install dependencies:

```
cd hello-prisma
```

Initialize Prisma in your project:

Update your `prisma/schema.prisma`:

```
generator client {
  provider = "prisma-client"
  output   = "./generated"
}

datasource db {
  provider = "postgresql"
}

model User {
  id    Int     @id @default(autoincrement())
  email String  @unique
  name  String?
  posts Post[]
}

model Post {
  id        Int      @id @default(autoincrement())
  title     String
  content   String?
  published Boolean  @default(false)
  author    User?    @relation(fields: [authorId], references: [id])
  authorId  Int?
}
```

Create a `prisma.config.ts` file in the root of your project:

```
import { defineConfig, env } from "prisma/config";
import "dotenv/config";

export default defineConfig({
  schema: "prisma/schema.prisma",
  migrations: {
    path: "prisma/migrations",
    seed: "tsx ./prisma/seed.ts",
  },
  datasource: {
    url: env("DATABASE_URL"),
  },
});
```

Update your `.env` file with your database connection string:

```
DATABASE_URL="postgresql://user:password@localhost:5432/mydb"
```

Run the migration to create your database tables:

Create `server/utils/db.ts`. Nuxt auto-imports exports from `server/utils`, making `prisma` available in all API routes:

```
import { PrismaPg } from "@prisma/adapter-pg";
import { PrismaClient } from "../../prisma/generated/client";

const prismaClientSingleton = () => {
  const pool = new PrismaPg({ connectionString: process.env.DATABASE_URL! });
  return new PrismaClient({ adapter: pool });
};

type PrismaClientSingleton = ReturnType<typeof prismaClientSingleton>;

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClientSingleton | undefined;
};

export const prisma = globalForPrisma.prisma ?? prismaClientSingleton();

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
```

Create an API route to fetch users. The `prisma` instance is auto-imported:

```
export default defineEventHandler(async () => {
  const users = await prisma.user.findMany({
    include: { posts: true },
  });
  return users;
});
```

Create an API route to create a user:

```
export default defineEventHandler(async (event) => {
  const body = await readBody<{ name: string; email: string }>(event);

  const user = await prisma.user.create({
    data: {
      name: body.name,
      email: body.email,
    },
  });

  return user;
});
```

## [5\. Create a page](https://www.prisma.io/docs/guides/frameworks/nuxt#5-create-a-page)

Update `app.vue` to display users:

```
<template>
  <div>
    <h1>Users</h1>
    <ul v-if="users?.length">
      <li v-for="user in users" :key="user.id">{{ user.name }} ({{ user.email }})</li>
    </ul>
    <p v-else>No users yet.</p>
  </div>
</template>

<script setup>
  const { data: users } = await useFetch('/api/users')
</script>
```

Start the development server:

Open `http://localhost:3000` to see your app.

Create a seed file to populate your database with sample data:

```
import "dotenv/config";
import { PrismaClient } from "./generated/client";
import { PrismaPg } from "@prisma/adapter-pg";

const adapter = new PrismaPg({ connectionString: process.env.DATABASE_URL! });
const prisma = new PrismaClient({ adapter });

async function main() {
  const alice = await prisma.user.create({
    data: {
      name: "Alice",
      email: "alice@prisma.io",
      posts: {
        create: { title: "Hello World", published: true },
      },
    },
  });
  console.log(`Created user: ${alice.name}`);
}

main()
  .then(() => prisma.$disconnect())
  .catch(async (e) => {
    console.error(e);
    await prisma.$disconnect();
    process.exit(1);
  });
```

Run the seed:

You can deploy your Nuxt application to Vercel using one of two methods:

### [Option A: Deploy using Vercel CLI](https://www.prisma.io/docs/guides/frameworks/nuxt#option-a-deploy-using-vercel-cli)

1.  Install the Vercel CLI (if not already installed):
    
2.  Deploy your project:
    
3.  Set the `DATABASE_URL` environment variable:
    
    -   Go to your [Vercel Dashboard](https://vercel.com/dashboard)
    -   Select your project
    -   Navigate to **Settings** → **Environment Variables**
    -   Add `DATABASE_URL` with your database connection string
4.  Redeploy your application to apply the environment variable:
    

### [Option B: Deploy using Git integration](https://www.prisma.io/docs/guides/frameworks/nuxt#option-b-deploy-using-git-integration)

1.  Push your code to a Git repository (GitHub, GitLab, or Bitbucket).
    
2.  Add `prisma generate` to your `postinstall` script in `package.json` to ensure Prisma Client is generated during deployment:
    
    ```
    {
      "scripts": {
        "postinstall": "prisma generate",
        "build": "nuxt build",
        "dev": "nuxt dev"
      }
    }
    ```
    
3.  Import your project in Vercel:
    
    -   Go to [Vercel Dashboard](https://vercel.com/dashboard)
    -   Click **Add New** → **Project**
    -   Import your Git repository
    -   Vercel will automatically detect it as a Nuxt project
4.  Configure environment variables:
    
    -   Before deploying, go to **Environment Variables**
    -   Add `DATABASE_URL` with your database connection string
    -   Click **Deploy**

Vercel will automatically build and deploy your Nuxt application. The deployment process is the same as any other Node.js application, and Prisma Client will be generated during the build process thanks to the `postinstall` script.

-   Explore the [full Nuxt + Prisma example](https://github.com/prisma/prisma-examples/tree/latest/orm/nuxt) for a complete blog application
-   Learn about [Prisma Client API](https://www.prisma.io/docs/orm/prisma-client/setup-and-configuration/introduction)
-   Set up [Prisma Postgres](https://www.prisma.io/docs/postgres) for a managed database

[Edit on GitHub](https://github.com/prisma/docs/edit/main/apps/docs/content/docs/guides/frameworks/nuxt.mdx)