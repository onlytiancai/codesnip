import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Static, ListView, ListItem, Button, Input, RichLog, TextArea
from textual.binding import Binding
from textual.screen import Screen
from textual import events
from rich.console import Console
from rich.markdown import Markdown

log_handler = TimedRotatingFileHandler(
    'rss_reader.log',
    when='midnight',
    interval=1,
    backupCount=7,
    encoding='utf-8'
)
log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
log_handler.setLevel(logging.DEBUG)

root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)
root_logger.addHandler(log_handler)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.propagate = False

from .models.database import Database
from .models.feed import Feed, Article
from .services.fetcher import FeedFetcher
from .services.parser import FeedParser
from .services.html2md import Html2Md


class AddFeedModal(Screen):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("enter", "submit", "Add", priority=True),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("Add Feed URL:", id="label")
            yield Input(placeholder="https://example.com/feed.xml", id="feed_url")
            yield Static("[yellow]Enter[/yellow] to add  |  [yellow]Esc[/yellow] to cancel", markup=True, id="hint")

    def action_cancel(self) -> None:
        log.info("AddFeedModal: Escape pressed")
        self.app.pop_screen()

    def action_submit(self) -> None:
        url = self.query_one("#feed_url", Input).value
        log.info(f"AddFeedModal: Submit with URL: {url}")
        asyncio.create_task(self.app.add_feed(url))


class FeedListScreen(Screen):
    BINDINGS = [
        Binding("a", "add_feed", "Add Feed", priority=True),
        Binding("d", "delete_feed", "Delete", priority=True),
        Binding("j", "list_down", "Down", priority=True),
        Binding("k", "list_up", "Up", priority=True),
        Binding("down", "list_down", "Down"),
        Binding("up", "list_up", "Up"),
        Binding("right", "enter_article_list", "Enter"),
        Binding("enter", "enter_article_list", "Enter"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("RSS Feeds", id="title")
        yield Static("", id="loading_msg")
        yield ListView(id="feed_list")
        yield Footer()

    def on_mount(self) -> None:
        log.info("FeedListScreen: Mounted, refreshing feeds")
        self.update_loading()
        asyncio.create_task(self.refresh_feeds())

    def on_screen_resume(self) -> None:
        feed_list = self.query_one("#feed_list", ListView)
        if feed_list.children:
            if feed_list.index is None:
                feed_list.index = 0
        self.call_later(lambda: self.query_one("#feed_list", ListView).focus())

    def update_loading(self):
        app = self.app
        loading_el = self.query_one("#loading_msg", Static)
        if app.is_loading:
            loading_el.update(f"[yellow]Loading: {app.loading_message or 'Please wait...'}[/yellow]")
        else:
            loading_el.update("")

    async def refresh_feeds(self):
        app = self.app
        log.debug(f"FeedListScreen.refresh_feeds: app.feeds = {len(app.feeds)} feeds")
        await asyncio.sleep(0.1)
        feed_list = self.query_one("#feed_list", ListView)
        feed_list.clear()
        log.debug("FeedListScreen: Cleared feed list")

        if not app.feeds:
            log.debug("FeedListScreen: No feeds, showing empty state message")
            empty_item = ListItem(Static("[empty] No feeds. Press 'a' to add one."))
            empty_item.key = "empty"
            feed_list.append(empty_item)
        else:
            log.debug(f"FeedListScreen: About to add {len(app.feeds)} feeds to list")
            for feed in app.feeds:
                articles = app.articles_by_feed.get(feed.id, [])
                total_count = len(articles)
                unread_count = len([a for a in articles if not a.is_read])
                suffix = f" ({total_count}/{unread_count})" if unread_count > 0 else f" ({total_count})"
                item = ListItem(Static(f"{feed.title}{suffix}"))
                item.key = str(feed.id)
                feed_list.append(item)
                log.debug(f"FeedListScreen: Added feed to list: {feed.title} (id={feed.id})")
            log.debug("FeedListScreen: Done adding feeds to list")
            feed_list.index = 0

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        if event.item and event.item.key and event.item.key != "empty":
            feed_id = int(event.item.key)
            self.app.current_feed_id = feed_id
            self.app.push_screen(ArticleListScreen(self.app, feed_id))

    def action_enter_article_list(self) -> None:
        feed_list = self.query_one("#feed_list", ListView)
        if feed_list.index is not None and feed_list.index >= 0:
            try:
                item = feed_list.children[feed_list.index]
                if hasattr(item, 'key') and item.key and item.key != "empty":
                    feed_id = int(item.key)
                    self.app.current_feed_id = feed_id
                    self.app.push_screen(ArticleListScreen(self.app, feed_id))
            except (IndexError, AttributeError):
                pass

    def action_add_feed(self) -> None:
        self.app.push_screen(AddFeedModal())

    def action_delete_feed(self) -> None:
        if self.app.current_feed_id:
            asyncio.create_task(self.app.action_delete_feed())

    def action_list_up(self) -> None:
        feed_list = self.query_one("#feed_list", ListView)
        if feed_list.index is not None and feed_list.index > 0:
            feed_list.index -= 1

    def action_list_down(self) -> None:
        feed_list = self.query_one("#feed_list", ListView)
        if feed_list.index is not None and feed_list.index < len(feed_list.children) - 1:
            feed_list.index += 1


class ArticleListScreen(Screen):
    BINDINGS = [
        Binding("left", "go_back", "Back"),
        Binding("b", "go_back", "Back"),
        Binding("j", "list_down", "Down", priority=True),
        Binding("k", "list_up", "Up", priority=True),
        Binding("down", "list_down", "Down"),
        Binding("up", "list_up", "Up"),
        Binding("right", "enter_article_detail", "Enter"),
        Binding("enter", "enter_article_detail", "Enter"),
        Binding("r", "toggle_read", "Read"),
        Binding("s", "toggle_star", "Star"),
    ]

    def __init__(self, app: "RSSReaderApp", feed_id: int):
        super().__init__()
        self._feed_id = feed_id

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Articles", id="title")
        yield ListView(id="article_list")
        yield Footer()

    async def on_mount(self) -> None:
        await self.refresh_articles()

    def on_screen_resume(self) -> None:
        self.call_later(self._ensure_selection)

    def _ensure_selection(self):
        article_list = self.query_one("#article_list", ListView)
        if article_list.children:
            if article_list.index is None:
                article_list.index = 0
            article_list.refresh()
        article_list.focus()

    async def refresh_articles(self):
        app = self.app
        feed_id = self._feed_id
        app.current_feed_id = feed_id
        feed = await app.db.get_feed(feed_id)
        title = feed.title if feed else "Articles"
        self.query_one("#title", Static).update(title)

        article_list = self.query_one("#article_list", ListView)
        article_list.clear()
        for article in app.articles_by_feed.get(feed_id, []):
            prefix = "[yellow]★[/yellow] " if article.is_starred else ("[green]●[/green] " if article.is_read else "")
            published_short = article.published[:10] if article.published else ""
            title_line = f"{prefix}{article.title}"
            item = ListItem(Static(f"{title_line} [dim]{published_short}[/dim]", markup=True))
            item.key = str(article.id)
            article_list.append(item)
        if article_list.children:
            article_list.index = 0

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        if event.item and event.item.key:
            article_id = int(event.item.key)
            self.app.current_article_id = article_id
            asyncio.create_task(self._enter_detail_and_mark_read(article_id))

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_enter_article_detail(self) -> None:
        article_list = self.query_one("#article_list", ListView)
        if article_list.index is not None and article_list.index >= 0:
            try:
                item = article_list.children[article_list.index]
                if hasattr(item, 'key') and item.key:
                    article_id = int(item.key)
                    self.app.current_article_id = article_id
                    asyncio.create_task(self._enter_detail_and_mark_read(article_id))
            except (IndexError, AttributeError):
                pass

    async def _enter_detail_and_mark_read(self, article_id: int) -> None:
        await self.app.mark_article_read(article_id)
        await self.refresh_articles()
        self.app.push_screen(ArticleDetailScreen(self.app, article_id))

    async def action_toggle_read(self) -> None:
        article_list = self.query_one("#article_list", ListView)
        if article_list.index is not None and article_list.index >= 0:
            try:
                item = article_list.children[article_list.index]
                if hasattr(item, 'key') and item.key:
                    article_id = int(item.key)
                    await self.app.toggle_read(article_id)
                    await self.refresh_articles()
            except (IndexError, AttributeError):
                pass

    async def action_toggle_star(self) -> None:
        article_list = self.query_one("#article_list", ListView)
        if article_list.index is not None and article_list.index >= 0:
            try:
                item = article_list.children[article_list.index]
                if hasattr(item, 'key') and item.key:
                    article_id = int(item.key)
                    await self.app.toggle_star(article_id)
                    await self.refresh_articles()
            except (IndexError, AttributeError):
                pass
            except (IndexError, AttributeError):
                pass

    def action_list_up(self) -> None:
        article_list = self.query_one("#article_list", ListView)
        if article_list.index is not None and article_list.index > 0:
            article_list.index -= 1

    def action_list_down(self) -> None:
        article_list = self.query_one("#article_list", ListView)
        if article_list.index is not None and article_list.index < len(article_list.children) - 1:
            article_list.index += 1


class ArticleDetailScreen(Screen):
    BINDINGS = [
        Binding("c", "copy_selection", "Copy"),
        Binding("b", "go_back", "Back"),
    ]

    def __init__(self, app: "RSSReaderApp", article_id: int):
        super().__init__()
        self._article_id = article_id

    def compose(self) -> ComposeResult:
        yield Header()
        yield TextArea(id="content", read_only=True, show_cursor=False)
        yield Footer()

    def on_mount(self) -> None:
        asyncio.create_task(self.load_article())

    async def load_article(self):
        app = self.app
        article_id = self._article_id
        article = await app.db.get_article(article_id)
        if not article:
            log.warning(f"ArticleDetailScreen: Article id={article_id} not found")
            return

        if article.content_md:
            content = article.content_md
        elif article.description:
            content = article.description
        else:
            content = article.content or "No content available."

        log.debug(f"ArticleDetailScreen: Loaded article id={article_id}: {article.title[:50]}...")

        text_area = self.query_one("#content", TextArea)
        text_area.clear()
        text_area.text = content
        text_area.scroll_home()

    def action_copy_selection(self) -> None:
        text_area = self.query_one("#content", TextArea)
        selected = text_area.selected_text
        if selected:
            self.app.copy_to_clipboard(selected)
            self.notify("Copied to clipboard")

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def on_key(self, event: events.Key) -> None:
        if event.key == "o":
            asyncio.create_task(self._open_link())

    async def _open_link(self):
        app = self.app
        article_id = self._article_id
        article = await app.db.get_article(article_id)
        if article and article.link:
            import webbrowser
            webbrowser.open(article.link)


class RSSReaderApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #title {
        height: auto;
        padding: 1;
        text-style: bold;
        background: $surface;
    }
    ListView {
        height: 1fr;
    }
    ListItem {
        padding: 0 1;
    }
    #content {
        height: 1fr;
        max-width: 80;
        margin: 1 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+r", "refresh_all", "Refresh"),
    ]

    def __init__(self):
        super().__init__()
        self.db = Database()
        self.fetcher = FeedFetcher()
        self.parser = FeedParser()
        self.html2md = Html2Md()
        self.feeds: list[Feed] = []
        self.articles_by_feed: dict[int, list[Article]] = {}
        self.current_feed_id: int | None = None
        self.current_article_id: int | None = None
        self.is_loading = False
        self.loading_message = ""

    def on_mount(self) -> None:
        asyncio.create_task(self.init_db())

    async def init_db(self):
        log.info("init_db: Connecting to database...")
        await self.db.connect()
        log.info("init_db: Database connected, loading feeds...")
        await self.load_feeds()
        log.info(f"init_db: Loaded {len(self.feeds)} feeds")
        self.push_screen(FeedListScreen())
        log.info("init_db: FeedListScreen pushed")

    async def load_feeds(self):
        self.feeds = await self.db.get_feeds()
        for feed in self.feeds:
            articles = await self.db.get_articles(feed.id)
            self.articles_by_feed[feed.id] = articles

    async def add_feed(self, url: str):
        log.info(f"add_feed: Starting to add feed from URL: {url}")
        self.is_loading = True
        self.loading_message = f"Adding {url}..."

        try:
            xml = await self.fetcher.fetch(url)
            if not xml:
                log.error(f"add_feed: Failed to fetch XML from {url}")
                self.notify("Failed to fetch feed - check network or URL", severity="error")
                self.is_loading = False
                self.loading_message = ""
                return

            log.debug(f"add_feed: Fetched XML, length = {len(xml)} bytes")

            feed = self.parser.parse_feed(url, xml)
            if not feed:
                log.error(f"add_feed: Failed to parse feed from {url}")
                self.notify("Failed to parse feed - invalid RSS/Atom", severity="error")
                self.is_loading = False
                self.loading_message = ""
                return

            log.info(f"add_feed: Parsed feed title: {feed.title}")

            feed_id = await self.db.add_feed(feed)
            log.debug(f"add_feed: add_feed returned feed_id={feed_id}")

            if feed_id == -1:
                log.warning(f"add_feed: Feed already exists: {url}")
                self.notify(f"Feed already exists: {feed.title}", severity="warning")
                self.is_loading = False
                self.loading_message = ""
                self.pop_screen()
                return

            feed.id = feed_id
            log.debug(f"add_feed: Feed saved to DB with id={feed_id}")

            articles = self.parser.parse_articles(feed_id, xml)
            log.info(f"add_feed: Found {len(articles)} articles")

            for article in articles:
                article.content_md = self.html2md.convert(article.content or article.description or "")
                article_id = await self.db.add_article(article)
                log.debug(f"add_feed: Added article id={article_id}: {article.title[:50]}...")

            self.articles_by_feed[feed_id] = await self.db.get_articles(feed_id)
            self.feeds = await self.db.get_feeds()

            log.info(f"add_feed: Successfully added feed '{feed.title}' with {len(articles)} articles")

            self.is_loading = False
            self.loading_message = ""
            self.pop_screen()
            self.push_screen(FeedListScreen())
            self.notify(f"Added: {feed.title} ({len(articles)} articles)")

        except Exception as e:
            log.error(f"add_feed: Exception: {type(e).__name__}: {e}", exc_info=True)
            self.notify(f"Error adding feed: {e}", severity="error")
            self.is_loading = False
            self.loading_message = ""
            try:
                self.pop_screen()
            except:
                pass

    async def delete_feed(self, feed_id: int):
        await self.db.delete_feed(feed_id)
        if feed_id in self.articles_by_feed:
            del self.articles_by_feed[feed_id]
        self.feeds = await self.db.get_feeds()

    async def refresh_feed(self, feed_id: int):
        feed = await self.db.get_feed(feed_id)
        if not feed:
            return

        xml = await self.fetcher.fetch(feed.url)
        if not xml:
            self.notify("Failed to fetch feed", severity="error")
            return

        updated_feed = self.parser.parse_feed(feed.url, xml)
        if updated_feed:
            updated_feed.id = feed_id
            await self.db.update_feed(updated_feed)

        articles = self.parser.parse_articles(feed_id, xml)
        for article in articles:
            article.content_md = self.html2md.convert(article.content or article.description or "")
            await self.db.add_article(article)

        self.articles_by_feed[feed_id] = await self.db.get_articles(feed_id)
        self.feeds = await self.db.get_feeds()

    async def mark_article_read(self, article_id: int):
        await self.db.mark_article_read(article_id, True)
        for feed_id, articles in self.articles_by_feed.items():
            for article in articles:
                if article.id == article_id:
                    article.is_read = True
                    break

    async def toggle_read(self, article_id: int):
        article = await self.db.get_article(article_id)
        if article:
            article.is_read = not article.is_read
            await self.db.update_article(article)
            if self.current_feed_id and self.current_feed_id in self.articles_by_feed:
                for a in self.articles_by_feed[self.current_feed_id]:
                    if a.id == article_id:
                        a.is_read = article.is_read
                        break

    async def toggle_star(self, article_id: int):
        article = await self.db.get_article(article_id)
        if article:
            article.is_starred = not article.is_starred
            await self.db.update_article(article)
            if self.current_feed_id and self.current_feed_id in self.articles_by_feed:
                for a in self.articles_by_feed[self.current_feed_id]:
                    if a.id == article_id:
                        a.is_starred = article.is_starred
                        break

    def action_add_feed(self) -> None:
        log.info("action_add_feed: Opening add feed modal")
        self.push_screen(AddFeedModal())

    async def action_delete_feed(self) -> None:
        if not self.current_feed_id:
            self.notify("Select a feed first", severity="warning")
            return
        await self.delete_feed(self.current_feed_id)
        self.current_feed_id = None
        self.app.pop_screen()
        self.push_screen(FeedListScreen())

    async def action_refresh_all(self) -> None:
        log.info("action_refresh_all: Refreshing all feeds")
        for feed in self.feeds:
            await self.refresh_feed(feed.id)
        self.notify("All feeds refreshed")
        self.push_screen(FeedListScreen())

    def action_quit(self) -> None:
        if self.is_loading:
            log.warning("action_quit: Cannot quit while loading")
            self.notify(f"Cannot quit: {self.loading_message or 'Loading in progress...'}", severity="warning")
            return
        log.info("action_quit: Quitting app")
        asyncio.create_task(self.db.close())
        self.exit()

    def copy_to_clipboard(self, text: str) -> None:
        import pyperclip
        pyperclip.copy(text)