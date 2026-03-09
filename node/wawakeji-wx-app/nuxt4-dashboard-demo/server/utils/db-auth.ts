import { prisma } from "./db";

// Use bcrypt for password hashing
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

async function createUser(data: { email: string; name?: string; password?: string; avatar?: string }) {
  return prisma.user.create({
    data,
    include: { accounts: true },
  });
}

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