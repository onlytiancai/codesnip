#!/usr/bin/env python3
"""
Scrape a rendered page via CDP, save HTML to temp, convert to markdown via defuddle.

Usage:
    python tools/scrape_page.py "https://example.com" "output.md"
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import websockets


def find_chrome_debugger_url() -> str | None:
    """Get the first available Chrome CDP websocket URL (skip DevTools itself)."""
    try:
        with urllib.request.urlopen("http://localhost:9222/json", timeout=5) as resp:
            targets = json.loads(resp.read())
        for target in targets:
            url = target.get("url", "")
            if url.startswith("devtools://"):
                continue
            if ws_url := target.get("webSocketDebuggerUrl"):
                return ws_url
    except Exception:
        pass
    return None


async def send_and_recv(ws, cmd: dict) -> dict:
    """Send a CDP command and wait for its response by matching ID."""
    await ws.send(json.dumps(cmd))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == cmd["id"]:
            return msg


async def navigate_and_get_html(ws_url: str, url: str) -> str:
    """Navigate to URL and get body outerHTML after page load."""
    async with websockets.connect(ws_url, max_size=None) as ws:
        # Enable Page domain
        await send_and_recv(ws, {"id": 1, "method": "Page.enable"})

        # Navigate
        await send_and_recv(ws, {
            "id": 2,
            "method": "Page.navigate",
            "params": {"url": url}
        })

        # Wait for load event
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("method") == "Page.loadEventFired":
                break

        # Wait extra for JS rendering
        await asyncio.sleep(3)

        # Get document
        resp = await send_and_recv(ws, {
            "id": 3,
            "method": "DOM.getDocument",
            "params": {"depth": 0}
        })
        root_node_id = resp["result"]["root"]["nodeId"]

        # QuerySelector for body
        resp = await send_and_recv(ws, {
            "id": 4,
            "method": "DOM.querySelector",
            "params": {"nodeId": root_node_id, "selector": "body"}
        })
        body_node_id = resp["result"]["nodeId"]

        # Get outerHTML
        resp = await send_and_recv(ws, {
            "id": 5,
            "method": "DOM.getOuterHTML",
            "params": {"nodeId": body_node_id}
        })
        return resp["result"]["outerHTML"]


def run_defuddle(html_path: str, output_path: str):
    """Convert HTML to markdown using defuddle."""
    cmd = [
        "npx", "defuddle", "parse",
        html_path,
        "--markdown",
        "-o", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"defuddle error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Saved markdown to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape rendered page via CDP, save as markdown")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("output", help="Output markdown path")
    args = parser.parse_args()

    ws_url = find_chrome_debugger_url()
    if not ws_url:
        print("Error: Could not find Chrome debugger. Is Chrome running with --remote-debugging-port=9222?", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to Chrome at {ws_url}")
    html = asyncio.run(navigate_and_get_html(ws_url, args.url))

    # Save HTML to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html)
        html_path = f.name

    print(f"Saved HTML to: {html_path}")

    # Convert to markdown
    run_defuddle(html_path, args.output)

    # Cleanup temp HTML
    os.unlink(html_path)


if __name__ == "__main__":
    main()
