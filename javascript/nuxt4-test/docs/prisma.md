link: 

- https://www.prisma.io/docs/guides/nuxt
- https://www.prisma.io/docs/postgres/database/local-development
- https://marketplace.visualstudio.com/items?itemName=Prisma.prisma


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

local Prisma Postgres

- vs code 插件 和 prisma studio 加载失败时，需要重新启动 npx prisma dev
- 看起来 local Prisma Postgres 不支持多进程使用，比如打开网站时 vs code 插件加载会失败
- migrate 时要用 http 方式连接，网站运行时要用 tcp 方式连接