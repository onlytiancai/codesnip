"""URL content extraction service."""
import httpx
import trafilatura
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from typing import Optional, Tuple
import re


class ExtractorService:
    """Service for extracting content from URLs."""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    async def fetch_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch content from a URL.
        Returns (html_content, error_message).
        """
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.text, None
        except httpx.TimeoutException:
            return None, "Request timed out"
        except httpx.HTTPStatusError as e:
            return None, f"HTTP error: {e.response.status_code}"
        except Exception as e:
            return None, f"Error fetching URL: {str(e)}"

    async def extract_content(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract main content from a URL.
        Returns (markdown_content, title, error_message).
        """
        html, error = await self.fetch_url(url)
        if error:
            return None, None, error

        try:
            # Extract main content using trafilatura
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_images=True,
                include_links=True,
                favor_precision=True
            )

            if not extracted:
                # Fallback to BeautifulSoup
                extracted = self._extract_with_beautifulsoup(html)

            if not extracted:
                return None, None, "Could not extract content from the page"

            # Get title
            title = self._extract_title(html)

            # Convert to Markdown
            markdown = self._to_markdown(extracted, html)

            return markdown, title, None

        except Exception as e:
            return None, None, f"Error extracting content: {str(e)}"

    def _extract_with_beautifulsoup(self, html: str) -> Optional[str]:
        """Fallback extraction using BeautifulSoup."""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Try to find main content
        main_content = (
            soup.find('article') or
            soup.find('main') or
            soup.find('div', class_=re.compile(r'content|article|post', re.I)) or
            soup.find('body')
        )

        if main_content:
            return main_content.get_text(separator='\n', strip=True)

        return None

    def _extract_title(self, html: str) -> Optional[str]:
        """Extract title from HTML."""
        soup = BeautifulSoup(html, 'html.parser')

        # Try og:title first
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content']

        # Try title tag
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)

        # Try h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)

        return None

    def _to_markdown(self, text: str, html: str) -> str:
        """Convert extracted text to Markdown."""
        # Try to get more structured content from HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Find main content area
        main_content = (
            soup.find('article') or
            soup.find('main') or
            soup.find('div', class_=re.compile(r'content|article|post', re.I)) or
            soup.find('body')
        )

        if main_content:
            # Convert to markdown
            markdown = md(str(main_content), heading_style="atx")
            # Clean up extra whitespace
            markdown = re.sub(r'\n{3,}', '\n\n', markdown)
            return markdown.strip()

        # Fallback to plain text
        return text