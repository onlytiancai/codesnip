import "dotenv/config";
import { PrismaBetterSQLite3 } from "@prisma/adapter-better-sqlite3";
import { PrismaClient } from "../../generated/dictionary/client";

const connectionString = "file:./prisma/dictionary.db";

const dictionaryClientSingleton = () => {
  const adapter = new PrismaBetterSQLite3({ url: connectionString });
  return new PrismaClient({ adapter });
};

type DictionaryClientSingleton = ReturnType<typeof dictionaryClientSingleton>;

const globalForDictionary = globalThis as unknown as {
  dictionary: DictionaryClientSingleton | undefined;
};

export const dictionaryDb = globalForDictionary.dictionary ?? dictionaryClientSingleton();

if (process.env.NODE_ENV !== "production") globalForDictionary.dictionary = dictionaryDb;