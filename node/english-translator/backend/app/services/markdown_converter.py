"""Markdown conversion utilities."""
from typing import List, Tuple
import re
from markdown_it import MarkdownIt


class MarkdownConverter:
    """Utilities for Markdown processing."""

    def __init__(self):
        self.md = MarkdownIt()

    def split_into_chunks(
        self,
        markdown: str,
        max_chunk_size: int = 2000,
        threshold: int = 3000
    ) -> List[Tuple[int, str]]:
        """
        Split markdown into chunks at safe boundaries.
        Returns list of (chunk_index, chunk_content).
        """
        if len(markdown) <= threshold:
            return [(0, markdown)]

        chunks = []
        lines = markdown.split('\n')
        current_chunk = []
        current_size = 0
        chunk_index = 0

        # Track if we're inside a code block
        in_code_block = False
        # Track if we're inside a table
        in_table = False

        for line in lines:
            # Check for code block boundaries
            if line.strip().startswith('```'):
                in_code_block = not in_code_block

            # Check for table boundaries (simple heuristic)
            if '|' in line and not in_code_block:
                if not in_table and re.match(r'^\s*\|', line):
                    in_table = True
            elif in_table and not line.strip():
                in_table = False

            line_size = len(line) + 1  # +1 for newline

            # Check if we should split
            should_split = (
                current_size + line_size > max_chunk_size and
                not in_code_block and
                not in_table and
                current_chunk and
                self._is_safe_boundary(line, current_chunk[-1] if current_chunk else '')
            )

            if should_split:
                # Save current chunk
                chunk_content = '\n'.join(current_chunk)
                chunks.append((chunk_index, chunk_content))
                chunk_index += 1
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Don't forget the last chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append((chunk_index, chunk_content))

        return chunks

    def _is_safe_boundary(self, current_line: str, previous_line: str) -> bool:
        """Check if this is a safe place to split."""
        current_stripped = current_line.strip()
        previous_stripped = previous_line.strip()

        # Safe to split before a header
        if current_stripped.startswith('#'):
            return True

        # Safe to split at empty line after a paragraph
        if not current_stripped and previous_stripped:
            return True

        # Safe to split before a list
        if re.match(r'^[-*+]\s', current_stripped):
            return True

        # Safe to split before a numbered list
        if re.match(r'^\d+\.\s', current_stripped):
            return True

        # Safe to split before a blockquote
        if current_stripped.startswith('>'):
            return True

        # Safe to split before a horizontal rule
        if re.match(r'^[-*_]{3,}$', current_stripped):
            return True

        return False

    def merge_translations(self, chunks: List[str]) -> str:
        """Merge translated chunks back together."""
        return '\n\n'.join(chunks)

    def extract_metadata(self, markdown: str) -> dict:
        """Extract metadata from markdown content."""
        metadata = {
            'word_count': len(markdown.split()),
            'char_count': len(markdown),
            'headings': [],
            'code_blocks': 0,
            'tables': 0,
            'images': 0,
            'links': 0
        }

        lines = markdown.split('\n')
        in_code_block = False

        for line in lines:
            # Count headings
            if line.strip().startswith('#'):
                heading_match = re.match(r'^(#+)\s+(.+)$', line)
                if heading_match:
                    metadata['headings'].append({
                        'level': len(heading_match.group(1)),
                        'text': heading_match.group(2)
                    })

            # Track code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                if not in_code_block:
                    metadata['code_blocks'] += 1

            # Count tables (simple heuristic)
            if '|' in line and not in_code_block:
                if re.match(r'^\s*\|.+\|\s*$', line):
                    metadata['tables'] = max(metadata['tables'], 1)

            # Count images
            metadata['images'] += len(re.findall(r'!\[.*?\]\(.*?\)', line))

            # Count links (excluding images)
            metadata['links'] += len(re.findall(r'(?<!!)\[.*?\]\(.*?\)', line))

        return metadata