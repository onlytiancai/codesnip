-- CreateTable
CREATE TABLE "ImportQueue" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "url" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "stage" TEXT NOT NULL DEFAULT 'init',
    "progress" INTEGER NOT NULL DEFAULT 0,
    "totalImages" INTEGER NOT NULL DEFAULT 0,
    "processedImages" INTEGER NOT NULL DEFAULT 0,
    "result" JSONB,
    "error" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- CreateTable
CREATE TABLE "ImportImage" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "queueId" INTEGER NOT NULL,
    "originalUrl" TEXT NOT NULL,
    "localPath" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "error" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "ImportImage_queueId_fkey" FOREIGN KEY ("queueId") REFERENCES "ImportQueue" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_Article" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "title" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "excerpt" TEXT,
    "cover" TEXT,
    "content" TEXT,
    "splitContent" JSONB,
    "status" TEXT NOT NULL DEFAULT 'draft',
    "difficulty" TEXT NOT NULL DEFAULT 'beginner',
    "views" INTEGER NOT NULL DEFAULT 0,
    "bookmarks" INTEGER NOT NULL DEFAULT 0,
    "publishAt" DATETIME,
    "metaTitle" TEXT,
    "metaDesc" TEXT,
    "categoryId" INTEGER,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "authorId" INTEGER NOT NULL,
    "importTaskId" INTEGER,
    CONSTRAINT "Article_authorId_fkey" FOREIGN KEY ("authorId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "Article_categoryId_fkey" FOREIGN KEY ("categoryId") REFERENCES "Category" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "Article_importTaskId_fkey" FOREIGN KEY ("importTaskId") REFERENCES "ImportQueue" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "new_Article" ("authorId", "bookmarks", "categoryId", "content", "cover", "createdAt", "difficulty", "excerpt", "id", "metaDesc", "metaTitle", "publishAt", "slug", "splitContent", "status", "title", "updatedAt", "views") SELECT "authorId", "bookmarks", "categoryId", "content", "cover", "createdAt", "difficulty", "excerpt", "id", "metaDesc", "metaTitle", "publishAt", "slug", "splitContent", "status", "title", "updatedAt", "views" FROM "Article";
DROP TABLE "Article";
ALTER TABLE "new_Article" RENAME TO "Article";
CREATE UNIQUE INDEX "Article_slug_key" ON "Article"("slug");
CREATE UNIQUE INDEX "Article_importTaskId_key" ON "Article"("importTaskId");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

-- CreateIndex
CREATE INDEX "ImportQueue_status_idx" ON "ImportQueue"("status");

-- CreateIndex
CREATE INDEX "ImportQueue_createdAt_idx" ON "ImportQueue"("createdAt");

-- CreateIndex
CREATE INDEX "ImportImage_queueId_idx" ON "ImportImage"("queueId");
