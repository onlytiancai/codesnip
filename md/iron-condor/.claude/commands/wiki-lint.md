Health-check the LLM Wiki for issues.

Usage: /wiki-lint

Follow the Lint Workflow defined in CLAUDE.md:

Structural checks (use Grep and Glob tools):
1. Orphan pages — wiki pages with no inbound [[wikilinks]] from other pages
2. Broken links — [[WikiLinks]] pointing to pages that don't exist
3. Missing entity pages — names referenced in 3+ pages but lacking their own page

Semantic checks (read and reason over page content):
4. Contradictions — claims that conflict between pages
5. Stale summaries — pages not updated after newer sources changed the picture
6. Data gaps — important questions the wiki can't answer; suggest specific sources to find

Output a structured markdown lint report. At the end, ask if the user wants it saved to wiki/lint-report.md.

Append to wiki/log.md: ## [today's date] lint | Wiki health check
