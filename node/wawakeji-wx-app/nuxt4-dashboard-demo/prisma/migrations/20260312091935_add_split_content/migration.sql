-- AlterTable
ALTER TABLE "Article" ADD COLUMN "splitContent" JSONB;

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_ReadingHistory" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
    "articleId" INTEGER NOT NULL,
    "progress" INTEGER NOT NULL DEFAULT 0,
    "readingTime" INTEGER NOT NULL DEFAULT 0,
    "lastReadAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" DATETIME,
    CONSTRAINT "ReadingHistory_articleId_fkey" FOREIGN KEY ("articleId") REFERENCES "Article" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "ReadingHistory_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_ReadingHistory" ("articleId", "completedAt", "id", "lastReadAt", "progress", "userId") SELECT "articleId", "completedAt", "id", "lastReadAt", "progress", "userId" FROM "ReadingHistory";
DROP TABLE "ReadingHistory";
ALTER TABLE "new_ReadingHistory" RENAME TO "ReadingHistory";
CREATE UNIQUE INDEX "ReadingHistory_userId_articleId_key" ON "ReadingHistory"("userId", "articleId");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
