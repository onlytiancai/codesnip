# Article Scraper

A Nuxt 4 application for scraping and managing articles from various web sources.

## Features

- Web article extraction using @extractus/article-extractor
- Markdown conversion with Turndown
- Admin authentication with bcryptjs
- Background job processing with BullMQ + Redis
- Prisma ORM with database support
- End-to-end testing with Playwright

## Tech Stack

- **Framework**: Nuxt 4
- **Database**: Prisma (SQLite/PostgreSQL)
- **Queue**: BullMQ + Redis
- **Auth**: nuxt-auth-utils
- **UI**: @nuxt/ui (Tailwind CSS)
- **Testing**: Vitest + Playwright

## Setup

```bash
# Install dependencies
pnpm install

# Generate Prisma client
pnpm prisma:generate

# Push database schema
pnpm prisma:dbpush

# Run development server
pnpm dev
```

## Scripts

| Command | Description |
|---------|-------------|
| `pnpm dev` | Start development server |
| `pnpm build` | Build for production |
| `pnpm generate` | Generate static site |
| `pnpm preview` | Preview production build |
| `pnpm test` | Run unit tests |
| `pnpm test:e2e` | Run Playwright e2e tests |
| `pnpm prisma:generate` | Generate Prisma client |
| `pnpm prisma:dbpush` | Push database schema |

## Admin Account

- Email: admin@example.com
- Password: admin123
