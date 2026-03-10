-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_Vocabulary" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
    "word" TEXT NOT NULL,
    "phonetic" TEXT,
    "definition" TEXT NOT NULL,
    "example" TEXT,
    "progress" INTEGER NOT NULL DEFAULT 0,
    "articleId" INTEGER,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "lastReviewAt" DATETIME,
    CONSTRAINT "Vocabulary_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "Vocabulary_articleId_fkey" FOREIGN KEY ("articleId") REFERENCES "Article" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "new_Vocabulary" ("articleId", "createdAt", "definition", "example", "id", "lastReviewAt", "phonetic", "progress", "userId", "word") SELECT "articleId", "createdAt", "definition", "example", "id", "lastReviewAt", "phonetic", "progress", "userId", "word" FROM "Vocabulary";
DROP TABLE "Vocabulary";
ALTER TABLE "new_Vocabulary" RENAME TO "Vocabulary";
CREATE UNIQUE INDEX "Vocabulary_userId_word_key" ON "Vocabulary"("userId", "word");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
