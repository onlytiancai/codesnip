link: https://www.prisma.io/docs/guides/nuxt


install

    pnpm add @prisma/client @prisma/adapter-pg pg
    pnpm add -D prisma @types/pg dotenv tsx

    npx prisma init

    npx prisma dev # 独立窗口
    # 按 h 复制 http DATABASE_URL 到 .env

    vi prisma/schema.prisma
    vi prisma.config.ts

    npx prisma migrate dev --name init  # 需要使用 http DATABASE_URL
    npx prisma generate

    vi server/utils/db.ts

    pnpm dev # 需要使用 tcp DATABASE_URL
    npx prisma studio # 需要使用 tcp DATABASE_URL
