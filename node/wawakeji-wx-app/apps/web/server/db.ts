import prismaPkg from '@prisma/client'
const { PrismaClient } = prismaPkg

const globalForPrisma = globalThis as unknown as {
  prisma: InstanceType<typeof PrismaClient> | undefined
}

export const prisma = globalForPrisma.prisma ?? new PrismaClient({
  datasourceUrl: process.env.DATABASE_URL || 'file:../../packages/database/prisma/dev.db',
})

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma
}
