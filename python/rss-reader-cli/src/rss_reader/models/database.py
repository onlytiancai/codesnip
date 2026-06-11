import aiosqlite
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

from .feed import Feed, Article


class Database:
    def __init__(self, db_path: str = "rss_reader.db"):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._create_tables()

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def _create_tables(self):
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                description TEXT,
                last_fetched TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feed_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                link TEXT NOT NULL,
                description TEXT,
                content TEXT,
                content_md TEXT,
                author TEXT,
                published TEXT,
                guid TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                FOREIGN KEY (feed_id) REFERENCES feeds(id) ON DELETE CASCADE,
                UNIQUE(feed_id, guid)
            )
        """)
        await self._conn.commit()

    async def add_feed(self, feed: Feed) -> int:
        try:
            cursor = await self._conn.execute(
                """INSERT INTO feeds (title, url, description, last_fetched, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (feed.title, feed.url, feed.description or "", feed.last_fetched or "", feed.created_at),
            )
            await self._conn.commit()
            return cursor.lastrowid
        except aiosqlite.IntegrityError as e:
            log.warning(f"Feed already exists: {feed.url}")
            return -1

    async def get_feeds(self) -> list[Feed]:
        cursor = await self._conn.execute("SELECT * FROM feeds ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [Feed(**dict(row)) for row in rows]

    async def get_feed(self, feed_id: int) -> Optional[Feed]:
        cursor = await self._conn.execute("SELECT * FROM feeds WHERE id = ?", (feed_id,))
        row = await cursor.fetchone()
        return Feed(**dict(row)) if row else None

    async def update_feed(self, feed: Feed):
        await self._conn.execute(
            """UPDATE feeds SET title=?, url=?, description=?, last_fetched=? WHERE id=?""",
            (feed.title, feed.url, feed.description or "", feed.last_fetched or "", feed.id),
        )
        await self._conn.commit()

    async def delete_feed(self, feed_id: int):
        await self._conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
        await self._conn.commit()

    async def add_article(self, article: Article) -> int:
        try:
            cursor = await self._conn.execute(
                """INSERT INTO articles
                   (feed_id, title, link, description, content, content_md, author, published, guid, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    article.feed_id, article.title, article.link,
                    article.description or "", article.content or "", article.content_md or "",
                    article.author or "", article.published or "",
                    article.guid, article.fetched_at,
                ),
            )
            await self._conn.commit()
            return cursor.lastrowid
        except aiosqlite.IntegrityError:
            return -1

    async def get_articles(self, feed_id: int) -> list[Article]:
        cursor = await self._conn.execute(
            "SELECT * FROM articles WHERE feed_id = ? ORDER BY published DESC",
            (feed_id,),
        )
        rows = await cursor.fetchall()
        return [Article(**dict(row)) for row in rows]

    async def get_article(self, article_id: int) -> Optional[Article]:
        cursor = await self._conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
        row = await cursor.fetchone()
        return Article(**dict(row)) if row else None

