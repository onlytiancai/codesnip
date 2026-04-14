Build the LLM Wiki knowledge graph.

Usage: /wiki-graph

First try running: python tools/build_graph.py --open

If that fails (missing dependencies), build the graph manually:

1. Use Grep to find all [[wikilinks]] across every file in wiki/
2. Build a nodes list: one node per wiki page, with id=relative-path, label=title, type from frontmatter
3. Build an edges list: one edge per [[wikilink]], tagged EXTRACTED
4. Infer additional implicit relationships between pages not captured by wikilinks — tag these INFERRED with a confidence score (0.0–1.0); tag low-confidence ones AMBIGUOUS
5. Write graph/graph.json with {nodes, edges, built: today}
6. Write graph/graph.html as a self-contained vis.js page (nodes colored by type, edges colored by type, interactive, searchable)

After building, summarize: node count, edge count, breakdown by type, and the most connected nodes (hubs).

Append to wiki/log.md: ## [today's date] graph | Knowledge graph rebuilt
