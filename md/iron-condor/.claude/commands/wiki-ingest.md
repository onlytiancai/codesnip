Ingest a source document into the LLM Wiki.

Usage: /wiki-ingest $ARGUMENTS

$ARGUMENTS should be the path to a file in raw/, e.g. `raw/articles/my-article.md`

Follow the Ingest Workflow defined in CLAUDE.md exactly:
1. Read the source file at the given path
2. Read wiki/index.md and wiki/overview.md for current context
3. Write wiki/sources/<slug>.md (source page format per CLAUDE.md)
4. Update wiki/index.md — add the new entry under Sources
5. Update wiki/overview.md — revise synthesis if warranted
6. Create/update entity pages (wiki/entities/) for key people, companies, projects
7. Create/update concept pages (wiki/concepts/) for key ideas and frameworks
8. Flag any contradictions with existing wiki content
9. Append to wiki/log.md: ## [today's date] ingest | <Title>

After completing all writes, summarize: what was added, which pages were created or updated, and any contradictions found.
