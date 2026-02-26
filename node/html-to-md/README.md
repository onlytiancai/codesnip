# HTML to Markdown Converter

A Node.js CLI tool that converts HTML to Markdown format. Supports fetching HTML from URLs or reading from local files.

## Features

- Fetch HTML content from a URL
- Read HTML from a local file
- Smart content extraction (removes navigation, ads, etc.)
- Convert HTML to clean Markdown format
- Interactive mode for guided input
- Customizable output

## Installation

```bash
pnpm install
```

## Usage

### From URL

```bash
# Convert URL and save to file
pnpm start -- -u https://example.com/article -o output.md

# Convert URL and print to stdout
pnpm start -- -u https://example.com/article
```

### From Local File

```bash
# Convert local file and save to file
pnpm start -- -f input.html -o output.md

# Convert local file and print to stdout
pnpm start -- -f input.html
```

### Interactive Mode

```bash
pnpm start
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-u, --url <url>` | URL to fetch HTML from |
| `-f, --file <path>` | Local HTML file path |
| `-o, --output <path>` | Output Markdown file path |
| `-r, --readability` | Use @mozilla/readability for content extraction |

## Project Structure

```
html-to-md/
├── src/
│   ├── index.js        # Main entry point & CLI
│   ├── fetcher.js      # HTML fetching with undici
│   ├── extractor.js    # Content extraction
│   └── converter.js    # HTML to Markdown conversion
├── package.json
└── README.md
```

## How It Works

1. **Fetch/Read**: Fetches HTML from a URL using `undici` or reads from a local file
2. **Extract**: Uses `@extractus/article-extractor` to intelligently extract main content
   - Optional: Use `@mozilla/readability` with `-r` flag for alternative extraction
3. **Convert**: Converts HTML to Markdown using `turndown` with GFM plugin (tables, task lists, etc.)
4. **Output**: Saves to file or prints to stdout

## Examples

### Convert a blog post

```bash
pnpm start -- -u https://dev.to/some-article -o article.md
```

### Convert saved HTML

```bash
pnpm start -- -f downloaded-page.html -o notes.md
```

### Use Readability for extraction

```bash
# Use @mozilla/readability instead of article-extractor
pnpm start -- -u https://example.com/article -r -o article.md
```

## License

ISC
