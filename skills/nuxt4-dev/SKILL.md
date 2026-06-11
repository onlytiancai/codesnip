---
name: nuxt4-dev
description: |
  Full-stack Nuxt 4 development skill. Use this skill whenever working on a Nuxt 4 project needing help with:
  - Project setup and configuration (Nuxt 4.x, nuxt.config.ts, app/ directory structure)
  - Tailwind CSS v4 integration (@nuxt/ui v4.5+, CSS-first configuration)
  - Nuxt UI v3/v4 components (UTable with id/accessorKey, UModal with title/description props)
  - Prisma ORM setup (v6.x, better-sqlite3 adapter, migrations, singleton db.ts pattern)
  - Authentication with nuxt-auth-utils (v0.5.x, email/password, OAuth GitHub/Google)
  - Testing with Vitest (v4.x) and Playwright (v1.58+)
  - Admin dashboard patterns (role-based access, protected routes, data tables)

  Make sure to use this skill whenever the user mentions Nuxt, even if they just say "add a table", "set up login", or "create a dashboard" - this skill covers the complete workflow from initialization to deployment.
---

# Nuxt 4 Full-Stack Development Guide

This skill helps you develop full-stack applications with Nuxt 4, Tailwind CSS v4, Nuxt UI, Prisma ORM, and nuxt-auth-utils.

## Quick Start

When starting a new Nuxt 4 project or adding features, follow these steps:

1. **Project Setup** - Initialize Nuxt 4 with required modules
2. **Database Setup** - Configure Prisma ORM
3. **Authentication** - Set up nuxt-auth-utils for login/register/OAuth
4. **UI Components** - Use Nuxt UI v4 components correctly
5. **Testing** - Write unit and E2E tests

## Core Technologies

| Technology | Purpose | Key Files |
|------------|---------|-----------|
| Nuxt 4 | Full-stack framework | `nuxt.config.ts`, `app/`, `server/` |
| Tailwind CSS v4 | Styling | `assets/css/main.css` |
| Nuxt UI v3/v4 | UI components | `components/`, `pages/` |
| Prisma ORM | Database | `prisma/schema.prisma`, `server/utils/db.ts` |
| nuxt-auth-utils | Authentication | `server/api/auth/`, `server/utils/db-auth.ts` |
| Vitest | Unit testing | `test/unit/`, `vitest.config.ts` |
| Playwright | E2E testing | `test/e2e/`, `playwright.config.ts` |

## When to Use This Skill

Use this skill when:
- Setting up a new Nuxt 4 project with full-stack capabilities
- Configuring Tailwind CSS v4 in a Nuxt project
- Using Nuxt UI components (especially UTable, UModal)
- Setting up Prisma with SQLite/PostgreSQL
- Implementing authentication (email/password or OAuth)
- Writing tests for Nuxt applications
- Debugging common Nuxt 4 issues

## Reference Files

For detailed information on specific topics, read these reference files:

- **Project Structure** → `references/project-structure.md` - app/ directory, auto-imports, common pitfalls
- **Tailwind v4** → `references/tailwind4.md` - CSS configuration, new v4 features
- **Nuxt UI v4** → `references/nuxt-ui-v4.md` - Component usage, common pitfalls
- **Prisma** → `references/prisma.md` - Schema design, migrations, client setup
- **Testing** → `references/testing.md` - Vitest unit tests, Playwright E2E
- **Authentication** → `references/auth.md` - nuxt-auth-utils, OAuth setup

## Project Structure

```
project/
├── app/                    # Nuxt 4 app directory (replaces pages/, components/)
│   ├── assets/css/        # CSS files (main.css for Tailwind)
│   ├── components/        # Vue components
│   ├── composables/       # Composable functions
│   └── pages/             # Page components
├── server/
│   ├── api/               # API endpoints
│   ├── routes/            # Server routes (OAuth callbacks)
│   └── utils/             # Server utilities (db.ts, db-auth.ts)
├── prisma/
│   ├── schema.prisma      # Database schema
│   └── migrations/        # Database migrations
├── test/
│   ├── unit/              # Vitest unit tests
│   └── e2e/browser/       # Playwright E2E tests
├── nuxt.config.ts         # Nuxt configuration
├── vitest.config.ts       # Unit test config
├── vitest.e2e.config.ts   # E2E test config
└── playwright.config.ts   # Playwright config
```

## Common Workflows

### Adding a New Feature

1. Update `prisma/schema.prisma` with new models
2. Run `pnpm prisma migrate dev` to generate migration
3. Create API endpoints in `server/api/`
4. Create UI components in `app/components/` or `app/pages/`
5. Write tests in `test/unit/` or `test/e2e/`

### Debugging Common Issues

- **UTable column errors** → Check `references/nuxt-ui-v4.md` for `id` vs `key`
- **UModal accessibility warnings** → Add `title` and `description` props
- **Prisma client not found** → Run `pnpm prisma generate`
- **Auth session not working** → Check `server/utils/db-auth.ts` setup
- **Composable not defined** → Ensure composables are in `app/composables/`, remove manual imports, restart dev server
- **Failed to resolve import ~/composables/** → Move composables to `app/composables/`, delete `.nuxt/` and restart