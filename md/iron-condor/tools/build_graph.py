#!/usr/bin/env python3
"""
Build the knowledge graph from the wiki.

Usage:
    python tools/build_graph.py               # full rebuild
    python tools/build_graph.py --no-infer    # skip semantic inference (faster)
    python tools/build_graph.py --open        # open graph.html in browser after build

Outputs:
    graph/graph.json    — node/edge data (cached by SHA256)
    graph/graph.html    — interactive vis.js visualization

Edge types:
    EXTRACTED   — explicit [[wikilink]] in a page
    INFERRED    — Claude-detected implicit relationship
    AMBIGUOUS   — low-confidence inferred relationship
"""

import re
import json
import hashlib
import argparse
import webbrowser
from pathlib import Path
from datetime import date

import os

try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Community detection disabled. Run: pip install networkx")

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
GRAPH_DIR = REPO_ROOT / "graph"
GRAPH_JSON = GRAPH_DIR / "graph.json"
GRAPH_HTML = GRAPH_DIR / "graph.html"
CACHE_FILE = GRAPH_DIR / ".cache.json"
INFERRED_EDGES_FILE = GRAPH_DIR / ".inferred_edges.jsonl"
LOG_FILE = WIKI_DIR / "log.md"
SCHEMA_FILE = REPO_ROOT / "CLAUDE.md"

# Node type → color mapping
TYPE_COLORS = {
    "source": "#4CAF50",
    "entity": "#2196F3",
    "concept": "#FF9800",
    "synthesis": "#9C27B0",
    "unknown": "#9E9E9E",
}

EDGE_COLORS = {
    "EXTRACTED": "#555555",
    "INFERRED": "#FF5722",
    "AMBIGUOUS": "#BDBDBD",
}


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def call_llm(prompt: str, model_env: str, default_model: str, max_tokens: int = 4096) -> str:
    try:
        from litellm import completion
    except ImportError:
        print("Error: litellm not installed. Run: pip install litellm")
        import sys
        sys.exit(1)
        
    model = os.getenv(model_env, default_model)

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    response = completion(**kwargs)
    return response.choices[0].message.content


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def all_wiki_pages() -> list[Path]:
    return [p for p in WIKI_DIR.rglob("*.md")
            if p.name not in ("index.md", "log.md", "lint-report.md")]


def extract_wikilinks(content: str) -> list[str]:
    return list(set(re.findall(r'\[\[([^\]]+)\]\]', content)))


def extract_frontmatter_type(content: str) -> str:
    match = re.search(r'^type:\s*(\S+)', content, re.MULTILINE)
    return match.group(1).strip('"\'') if match else "unknown"


def page_id(path: Path) -> str:
    return path.relative_to(WIKI_DIR).as_posix().replace(".md", "")


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache: dict):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def build_nodes(pages: list[Path]) -> list[dict]:
    nodes = []
    for p in pages:
        content = read_file(p)
        node_type = extract_frontmatter_type(content)
        title_match = re.search(r'^title:\s*"?([^"\n]+)"?', content, re.MULTILINE)
        label = title_match.group(1).strip() if title_match else p.stem
        nodes.append({
            "id": page_id(p),
            "label": label,
            "type": node_type,
            "color": TYPE_COLORS.get(node_type, TYPE_COLORS["unknown"]),
            "path": str(p.relative_to(REPO_ROOT)),
        })
    return nodes


def build_extracted_edges(pages: list[Path]) -> list[dict]:
    """Pass 1: deterministic wikilink edges."""
    # Build a map from stem (lower) -> page_id for resolution
    stem_map = {p.stem.lower(): page_id(p) for p in pages}
    edges = []
    seen = set()
    for p in pages:
        content = read_file(p)
        src = page_id(p)
        for link in extract_wikilinks(content):
            target = stem_map.get(link.lower())
            if target and target != src:
                key = (src, target)
                if key not in seen:
                    seen.add(key)
                    edges.append({
                        "from": src,
                        "to": target,
                        "type": "EXTRACTED",
                        "color": EDGE_COLORS["EXTRACTED"],
                        "confidence": 1.0,
                    })
    return edges


def load_checkpoint() -> tuple[list[dict], set[str]]:
    """Load previously inferred edges from JSONL checkpoint file."""
    edges = []
    completed = set()
    if INFERRED_EDGES_FILE.exists():
        for line in INFERRED_EDGES_FILE.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                completed.add(record["page_id"])
                edges.extend(record.get("edges", []))
            except (json.JSONDecodeError, KeyError):
                continue
    return edges, completed


def append_checkpoint(page_id_str: str, edges: list[dict]):
    """Append one page's inferred edges to the JSONL checkpoint."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    record = {"page_id": page_id_str, "edges": edges, "ts": date.today().isoformat()}
    with open(INFERRED_EDGES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_inferred_edges(pages: list[Path], existing_edges: list[dict], cache: dict, resume: bool = True) -> list[dict]:
    """Pass 2: API-inferred semantic relationships with checkpoint/resume."""
    # Load checkpoint if resuming
    checkpoint_edges, completed_ids = ([], set())
    if resume:
        checkpoint_edges, completed_ids = load_checkpoint()
        if completed_ids:
            print(f"  checkpoint: {len(completed_ids)} pages already done, {len(checkpoint_edges)} edges loaded")

    new_edges = list(checkpoint_edges)

    # Only process pages that changed since last run
    changed_pages = []
    for p in pages:
        content = read_file(p)
        h = sha256(content)
        pid = page_id(p)
        if cache.get(str(p)) != h and pid not in completed_ids:
            changed_pages.append(p)
        else:
            # Page unchanged: load its inferred edges from cache perfectly
            src = page_id(p)
            for rel in entry.get("edges", []):
                new_edges.append({
                    "from": src,
                    "to": rel["to"],
                    "type": rel.get("type", "INFERRED"),
                    "title": rel.get("relationship", ""),
                    "label": "",
                    "color": EDGE_COLORS.get(rel.get("type", "INFERRED"), EDGE_COLORS["INFERRED"]),
                    "confidence": float(rel.get("confidence", 0.7)),
                })

    if not changed_pages:
        print("  no changed pages — skipping semantic inference")
        return new_edges

    total_pages = len(changed_pages)
    already_done = len(completed_ids)
    grand_total = total_pages + already_done
    print(f"  inferring relationships for {total_pages} remaining pages (of {grand_total} total)...")

    # Build a summary of existing nodes for context
    node_list = "\n".join(f"- {page_id(p)} ({extract_frontmatter_type(read_file(p))})" for p in pages)
    existing_edge_summary = "\n".join(
        f"- {e['from']} → {e['to']} (EXTRACTED)" for e in existing_edges[:30]
    )

    for i, p in enumerate(changed_pages, 1):
        content = read_file(p)[:2000]  # truncate for context efficiency
        src = page_id(p)
        global_idx = already_done + i
        print(f"    [{global_idx}/{grand_total}] Inferring for '{src}'... ", end="", flush=True)

        prompt = f"""Analyze this wiki page and identify implicit semantic relationships to other pages in the wiki.

Source page: {src}
Content:
{content}

All available pages:
{node_list}

Already-extracted edges from this page:
{existing_edge_summary}

Return ONLY a JSON object containing an "edges" array of NEW relationships not already captured by explicit wikilinks. The response must be STRICTLY valid JSON formatted exactly like this:
{{
  "edges": [
    {{"to": "page-id", "relationship": "one-line description", "confidence": 0.0-1.0, "type": "INFERRED or AMBIGUOUS"}}
  ]
}}

CRITICAL INSTRUCTION:
YOU MUST RETURN ONLY A RAW JSON STRING BEGINNING WITH {{ AND ENDING WITH }}. 
DO NOT OUTPUT BULLET POINTS. DO NOT OUTPUT MARKDOWN LISTS. 
ANY CONVERSATIONAL PREAMBLE WILL CAUSE A SYSTEM CRASH.

Rules:
- Only include pages from the available list above
- Confidence >= 0.7 → INFERRED, < 0.7 → AMBIGUOUS
- Do not repeat edges already in the extracted list
- Return {{"edges": []}} if no new relationships found
"""
        page_edges = []
        try:
            raw = call_llm(prompt, "LLM_MODEL_FAST", "claude-3-5-haiku-latest", max_tokens=1024)
            raw = raw.strip()
            
            # Robust JSON extraction
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)
            else:
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                
            inferred = json.loads(raw)
            edges_list = inferred.get("edges", [])
            for rel in edges_list:
                if isinstance(rel, dict) and "to" in rel:
                    edge = {
                        "from": src,
                        "to": rel["to"],
                        "type": rel.get("type", "INFERRED"),
                        "title": rel.get("relationship", ""),
                        "label": "",
                        "color": EDGE_COLORS.get(rel.get("type", "INFERRED"), EDGE_COLORS["INFERRED"]),
                        "confidence": rel.get("confidence", 0.7),
                    }
                    page_edges.append(edge)
                    new_edges.append(edge)
            print(f"-> Found {len(page_edges)} edges.")
        except json.JSONDecodeError as jde:
            print(f"-> [WARN] Invalid JSON: {str(jde)[:60]}")
        except Exception as e:
            err_msg = str(e).replace('\n', ' ')[:80]
            print(f"-> [ERROR] {err_msg}")
            import time
            time.sleep(2)

        # Persist checkpoint immediately after each page
        append_checkpoint(src, page_edges)

    return new_edges


def deduplicate_edges(edges: list[dict]) -> list[dict]:
    """Merge duplicate and bidirectional edges, keeping highest confidence."""
    best = {}  # (min(a,b), max(a,b)) -> edge
    for e in edges:
        a, b = e["from"], e["to"]
        key = (min(a, b), max(a, b))
        existing = best.get(key)
        if not existing or e.get("confidence", 0) > existing.get("confidence", 0):
            best[key] = e
    return list(best.values())


def detect_communities(nodes: list[dict], edges: list[dict]) -> dict[str, int]:
    """Assign community IDs to nodes using Louvain algorithm."""
    if not HAS_NETWORKX:
        return {}

    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"])
    for e in edges:
        G.add_edge(e["from"], e["to"])

    if G.number_of_edges() == 0:
        return {}

    try:
        communities = nx_community.louvain_communities(G, seed=42)
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i
        return node_to_community
    except Exception:
        return {}


COMMUNITY_COLORS = [
    "#E91E63", "#00BCD4", "#8BC34A", "#FF5722", "#673AB7",
    "#FFC107", "#009688", "#F44336", "#3F51B5", "#CDDC39",
]


def render_html(nodes: list[dict], edges: list[dict]) -> str:
    """Generate self-contained vis.js HTML with interactive filtering."""
    nodes_json = json.dumps(nodes, indent=2, ensure_ascii=False)
    edges_json = json.dumps(edges, indent=2, ensure_ascii=False)

    legend_items = "".join(
        f'<span style="background:{color};padding:3px 8px;margin:2px;border-radius:3px;font-size:12px">{t}</span>'
        for t, color in TYPE_COLORS.items() if t != "unknown"
    )

    n_extracted = len([e for e in edges if e.get('type') == 'EXTRACTED'])
    n_inferred = len([e for e in edges if e.get('type') == 'INFERRED'])
    n_ambiguous = len([e for e in edges if e.get('type') == 'AMBIGUOUS'])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Wiki — Knowledge Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin: 0; background: #1a1a2e; font-family: 'Inter', sans-serif; color: #eee; }}
  #graph {{ width: 100vw; height: 100vh; }}
  #controls {{
    position: fixed; top: 10px; left: 10px; background: rgba(10,10,30,0.88);
    padding: 14px; border-radius: 10px; z-index: 10; max-width: 280px;
    backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.08);
  }}
  #controls h3 {{ margin: 0 0 10px; font-size: 15px; letter-spacing: 0.5px; }}
  #search {{ width: 100%; padding: 6px 8px; margin-bottom: 10px; background: #222; color: #eee; border: 1px solid #444; border-radius: 6px; font-size: 13px; }}
  .filter-group {{ margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1); }}
  .filter-group label {{ display: block; font-size: 12px; color: #bbb; margin-bottom: 4px; }}
  .slider-row {{ display: flex; align-items: center; gap: 8px; margin-top: 4px; }}
  .slider-row input[type=range] {{ flex: 1; accent-color: #FF5722; }}
  .slider-val {{ font-size: 12px; color: #FF5722; min-width: 28px; text-align: right; font-weight: bold; }}
  .cb-row {{ display: flex; align-items: center; gap: 6px; font-size: 12px; margin: 3px 0; cursor: pointer; }}
  .cb-row input {{ accent-color: #FF5722; }}
  #info {{
    position: fixed; bottom: 10px; left: 10px; background: rgba(10,10,30,0.92);
    padding: 14px; border-radius: 10px; z-index: 10; max-width: 360px;
    display: none; backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.08);
  }}
  #info b {{ font-size: 14px; }}
  #stats {{ position: fixed; top: 10px; right: 10px; background: rgba(10,10,30,0.88); padding: 10px 14px; border-radius: 10px; font-size: 12px; backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.08); }}
</style>
</head>
<body>
<div id="controls">
  <h3>LLM Wiki Graph</h3>
  <input id="search" type="text" placeholder="Search nodes..." oninput="applyFilters()">
  <div>{legend_items}</div>
  <div class="filter-group">
    <label>Edge Types</label>
    <div class="cb-row"><input type="checkbox" id="cb-extracted" checked onchange="applyFilters()"><span style="color:#888">━</span> Extracted ({n_extracted})</div>
    <div class="cb-row"><input type="checkbox" id="cb-inferred" checked onchange="applyFilters()"><span style="color:#FF5722">━</span> Inferred ({n_inferred})</div>
    <div class="cb-row"><input type="checkbox" id="cb-ambiguous" onchange="applyFilters()"><span style="color:#BDBDBD">━</span> Ambiguous ({n_ambiguous})</div>
  </div>
  <div class="filter-group">
    <label>Min Confidence</label>
    <div class="slider-row">
      <input type="range" id="conf-slider" min="0" max="100" value="50" oninput="applyFilters()">
      <span class="slider-val" id="conf-val">0.50</span>
    </div>
  </div>
</div>
<div id="graph"></div>
<div id="info">
  <b id="info-title"></b><br>
  <span id="info-type" style="font-size:12px;color:#aaa"></span><br>
  <span id="info-path" style="font-size:11px;color:#666"></span><br>
  <span id="info-edges" style="font-size:11px;color:#888;margin-top:4px;display:block"></span>
</div>
<div id="stats"></div>
<script>
const allNodes = {nodes_json};
const allEdges = {edges_json};

const nodes = new vis.DataSet(allNodes);
const edges = new vis.DataSet(allEdges.map((e, i) => ({{ ...e, id: 'e'+i }})));

const container = document.getElementById("graph");
const network = new vis.Network(container, {{ nodes, edges }}, {{
  nodes: {{
    shape: "dot",
    size: 10,
    font: {{ color: "#ddd", size: 12, strokeWidth: 3, strokeColor: "#111" }},
    borderWidth: 1.5,
    scaling: {{ label: {{ drawThreshold: 9, maxVisible: 18 }} }},
  }},
  edges: {{
    width: 0.8,
    smooth: {{ type: "continuous" }},
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.4 }} }},
    color: {{ inherit: false }},
    hoverWidth: 2,
  }},
  physics: {{
    stabilization: {{ iterations: 200, updateInterval: 25 }},
    barnesHut: {{ gravitationalConstant: -3000, springLength: 200, springConstant: 0.02, damping: 0.12 }},
  }},
  interaction: {{ hover: true, tooltipDelay: 150, hideEdgesOnDrag: true, hideEdgesOnZoom: true }},
}});

network.on("click", params => {{
  if (params.nodes.length > 0) {{
    const nid = params.nodes[0];
    const node = nodes.get(nid);
    const connEdges = network.getConnectedEdges(nid);
    document.getElementById("info").style.display = "block";
    document.getElementById("info-title").textContent = node.label;
    document.getElementById("info-type").textContent = `Type: ${{node.type}} | Community: ${{node.group}}`;
    document.getElementById("info-path").textContent = node.path;
    document.getElementById("info-edges").textContent = `${{connEdges.length}} connections`;
  }} else {{
    document.getElementById("info").style.display = "none";
  }}
}});

network.on("hoverEdge", params => {{
  const edge = edges.get(params.edge);
  if (edge && edge.label) {{
    document.getElementById("info").style.display = "block";
    document.getElementById("info-title").textContent = `${{edge.from}} → ${{edge.to}}`;
    document.getElementById("info-type").textContent = `${{edge.type}} (confidence: ${{(edge.confidence || 0).toFixed(2)}})`;
    document.getElementById("info-path").textContent = edge.label || '';
    document.getElementById("info-edges").textContent = '';
  }}
}});

function applyFilters() {{
  const q = document.getElementById("search").value.toLowerCase();
  const showExtracted = document.getElementById("cb-extracted").checked;
  const showInferred = document.getElementById("cb-inferred").checked;
  const showAmbiguous = document.getElementById("cb-ambiguous").checked;
  const minConf = parseInt(document.getElementById("conf-slider").value) / 100;
  document.getElementById("conf-val").textContent = minConf.toFixed(2);

  // Filter edges
  let visibleNodes = new Set();
  let visibleEdgeCount = 0;
  edges.forEach(e => {{
    const typeOk = (e.type === 'EXTRACTED' && showExtracted) ||
                   (e.type === 'INFERRED' && showInferred) ||
                   (e.type === 'AMBIGUOUS' && showAmbiguous);
    const confOk = (e.confidence || 1.0) >= minConf;
    const show = typeOk && confOk;
    edges.update({{ id: e.id, hidden: !show }});
    if (show) {{
      visibleNodes.add(e.from);
      visibleNodes.add(e.to);
      visibleEdgeCount++;
    }}
  }});

  // Filter nodes by search + connectivity
  nodes.forEach(n => {{
    const searchOk = !q || n.label.toLowerCase().includes(q);
    const connected = visibleNodes.has(n.id);
    const show = searchOk && (connected || q);
    nodes.update({{ id: n.id, hidden: !show, opacity: show ? 1 : 0.1 }});
  }});

  document.getElementById("stats").textContent =
    `${{visibleNodes.size}} nodes · ${{visibleEdgeCount}} edges (filtered)`;
}}

// Initial stats
document.getElementById("stats").textContent =
  `${{nodes.length}} nodes · ${{edges.length}} edges`;

// Apply default filter (hide AMBIGUOUS by default)
setTimeout(() => applyFilters(), 500);
</script>
</body>
</html>"""


def append_log(entry: str):
    log_path = WIKI_DIR / "log.md"
    existing = read_file(log_path)
    log_path.write_text(entry.strip() + "\n\n" + existing, encoding="utf-8")


def build_graph(infer: bool = True, open_browser: bool = False, clean: bool = False):
    pages = all_wiki_pages()
    today = date.today().isoformat()

    if not pages:
        print("Wiki is empty. Ingest some sources first.")
        return

    print(f"Building graph from {len(pages)} wiki pages...")
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # Clean checkpoint if requested
    if clean and INFERRED_EDGES_FILE.exists():
        INFERRED_EDGES_FILE.unlink()
        print("  cleaned: removed inference checkpoint")

    cache = load_cache()

    # Pass 1: extracted edges
    print("  Pass 1: extracting wikilinks...")
    nodes = build_nodes(pages)
    edges = build_extracted_edges(pages)
    print(f"  → {len(edges)} extracted edges")

    # Pass 2: inferred edges
    if infer:
        print("  Pass 2: inferring semantic relationships...")
        inferred = build_inferred_edges(pages, edges, cache, resume=not clean)
        edges.extend(inferred)
        print(f"  → {len(inferred)} inferred edges")
        save_cache(cache)

    # Deduplicate edges
    before_dedup = len(edges)
    edges = deduplicate_edges(edges)
    if before_dedup != len(edges):
        print(f"  dedup: {before_dedup} → {len(edges)} edges")

    # Community detection
    print("  Running Louvain community detection...")
    communities = detect_communities(nodes, edges)
    for node in nodes:
        comm_id = communities.get(node["id"], -1)
        if comm_id >= 0:
            node["color"] = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
        node["group"] = comm_id

    # Save graph.json
    graph_data = {"nodes": nodes, "edges": edges, "built": today}
    GRAPH_JSON.write_text(json.dumps(graph_data, indent=2, ensure_ascii=False))
    print(f"  saved: graph/graph.json  ({len(nodes)} nodes, {len(edges)} edges)")

    # Save graph.html
    html = render_html(nodes, edges)
    GRAPH_HTML.write_text(html, encoding="utf-8")
    print(f"  saved: graph/graph.html")

    n_ext = len([e for e in edges if e['type']=='EXTRACTED'])
    n_inf = len([e for e in edges if e['type'] in ('INFERRED', 'AMBIGUOUS')])
    append_log(f"## [{today}] graph | Knowledge graph rebuilt\n\n{len(nodes)} nodes, {len(edges)} edges ({n_ext} extracted, {n_inf} inferred).")

    if open_browser:
        webbrowser.open(f"file://{GRAPH_HTML.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LLM Wiki knowledge graph")
    parser.add_argument("--no-infer", action="store_true", help="Skip semantic inference (faster)")
    parser.add_argument("--open", action="store_true", help="Open graph.html in browser")
    parser.add_argument("--clean", action="store_true", help="Delete checkpoint and force full re-inference")
    args = parser.parse_args()
    build_graph(infer=not args.no_infer, open_browser=args.open, clean=args.clean)
