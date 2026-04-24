from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Feed:
    id: Optional[int] = None
    title: str = ""
    url: str = ""
    description: Optional[str] = None
    last_fetched: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_fetched is None:
            self.last_fetched = ""


@dataclass
class Article:
    id: Optional[int] = None
    feed_id: int = 0
    title: str = ""
    link: str = ""
    description: Optional[str] = None
    content: Optional[str] = None
    content_md: Optional[str] = None
    author: Optional[str] = None
    published: Optional[str] = None
    guid: str = ""
    is_read: bool = False
    is_starred: bool = False
    fetched_at: Optional[str] = None

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.now().isoformat()
        if isinstance(self.is_read, int):
            self.is_read = bool(self.is_read)
        if isinstance(self.is_starred, int):
            self.is_starred = bool(self.is_starred)
