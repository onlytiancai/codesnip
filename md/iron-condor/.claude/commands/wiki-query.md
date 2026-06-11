Query the LLM Wiki and synthesize an answer.

Usage: /wiki-query $ARGUMENTS

$ARGUMENTS is the question to answer, e.g. `What are the main themes across all sources?`

Follow the Query Workflow defined in CLAUDE.md:
1. Read wiki/index.md to identify the most relevant pages
2. Read those pages (up to ~10 most relevant)
3. Synthesize a thorough markdown answer with [[PageName]] wikilink citations
4. Include a ## Sources section at the end listing pages you drew from
5. Ask the user if they want the answer saved as wiki/syntheses/<slug>.md

If the wiki is empty, say so and suggest running /wiki-ingest first.
