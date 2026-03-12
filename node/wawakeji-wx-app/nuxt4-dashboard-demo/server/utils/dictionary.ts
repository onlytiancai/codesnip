// Dictionary database utility - exported as a function to work with Nitro
import { PrismaBetterSQLite3 } from "@prisma/adapter-better-sqlite3"
import { PrismaClient } from "../../generated/dictionary/client"

let dictionaryClient: ReturnType<typeof PrismaClient> | null = null

export function useDictionaryDb() {
  if (!dictionaryClient) {
    const adapter = new PrismaBetterSQLite3({ url: "file:./prisma/dictionary.db" })
    dictionaryClient = new PrismaClient({ adapter })
  }
  return dictionaryClient
}