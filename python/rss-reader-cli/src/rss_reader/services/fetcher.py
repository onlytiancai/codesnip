import httpx
import logging
from typing import Optional

log = logging.getLogger(__name__)

class FeedFetcher:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    async def fetch(self, url: str) -> Optional[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
            except Exception as e:
                log.error(f"Failed to fetch {url}: {type(e).__name__}: {e}")
                return None
