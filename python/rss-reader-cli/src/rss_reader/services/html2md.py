from markdownify import markdownify as _convert


class Html2Md:
    def convert(self, html: str) -> str:
        if not html:
            return ""
        return _convert(html)