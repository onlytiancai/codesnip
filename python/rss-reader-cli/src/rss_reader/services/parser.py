import feedparser
from datetime import datetime
from typing import Optional

from ..models.feed import Feed, Article


class FeedParser:
    def parse_feed(self, url: str, xml_content: str) -> Optional[Feed]:
        parsed = feedparser.parse(xml_content)
        if not parsed.feed:
            return None

        feed_data = parsed.feed
        feed = Feed(
            title=feed_data.get("title", url),
            url=url,
            description=feed_data.get("description", ""),
            last_fetched=datetime.now().isoformat(),
        )
        return feed

    def parse_articles(self, feed_id: int, xml_content: str) -> list[Article]:
        parsed = feedparser.parse(xml_content)
        articles = []

        for entry in parsed.entries:
            published = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6]).isoformat()
                except Exception:
                    published = entry.get("published", "")

            # Try to get full content from content:encoded, fallback to content, then description
            content = ""
            if hasattr(entry, "content") and entry.content:
                content = entry.content[0].get("value", "") if entry.content else ""
            if not content and hasattr(entry, "content_encoded"):
                content = entry.content_encoded or ""
            if not content:
                content = self._get_description(entry)

            article = Article(
                feed_id=feed_id,
                title=entry.get("title", "Untitled"),
                link=entry.get("link", ""),
                description=self._get_description(entry),
                content=content,
                author=entry.get("author", ""),
                published=published,
                guid=entry.get("id", entry.get("link", "")),
            )
            articles.append(article)

        return articles

    def _get_description(self, entry) -> str:
        if hasattr(entry, "summary"):
            return entry.summary
        if hasattr(entry, "description"):
            return entry.description
        return ""
