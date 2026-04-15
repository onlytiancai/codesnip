#!/usr/bin/env node
/**
 * Scrape a rendered page via CDP, convert to markdown via defuddle.
 *
 * Dependencies:
 *   npm install ws jsdom defuddle
 *
 * Usage:
 *   node tools/scrape_page.mjs <url> [output.md]
 *   node tools/scrape_page.mjs <url>              # print to stdout
 *   node tools/scrape_page.mjs <url> -w ws://... # custom CDP WebSocket URL
 *   node tools/scrape_page.mjs <url> -t 5000     # wait 5s for JS rendering (default 3000)
 */

import { WebSocket } from "ws";
import { JSDOM } from "jsdom";
import { Defuddle } from "defuddle/node";
import { setTimeout as sleep } from "timers/promises";
import { writeFileSync, unlinkSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const DEFAULT_RENDER_TIMEOUT = 3000;

/**
 * Find the first Chrome CDP websocket URL, skipping DevTools itself.
 */
function findChromeDebuggerUrl(targets) {
  for (const target of targets) {
    if (target.url.startsWith("devtools://")) continue;
    if (target.webSocketDebuggerUrl) return target.webSocketDebuggerUrl;
  }
  return null;
}

/**
 * Send a CDP command and wait for its response by matching ID.
 */
async function sendAndRecv(ws, id, method, params = {}) {
  const cmd = { id, method, params };
  ws.send(JSON.stringify(cmd));

  return new Promise((resolve) => {
    ws.on("message", (data) => {
      const msg = JSON.parse(data.toString());
      if (msg.id === id) resolve(msg);
    });
  });
}

/**
 * Navigate to a URL via CDP and return document.body.outerHTML.
 */
async function navigateAndGetHtml(ws, url, renderTimeout) {
  // Enable Page domain
  await sendAndRecv(ws, 1, "Page.enable");

  // Navigate
  await sendAndRecv(ws, 2, "Page.navigate", { url });

  // Wait for load event
  await new Promise((resolve) => {
    ws.on("message", (data) => {
      const msg = JSON.parse(data.toString());
      if (msg.method === "Page.loadEventFired") resolve();
    });
  });

  // Extra wait for JS rendering
  await sleep(renderTimeout);

  // Get document
  const docResp = await sendAndRecv(ws, 3, "DOM.getDocument", { depth: 0 });
  const rootNodeId = docResp.result.root.nodeId;

  // QuerySelector for body
  const bodyResp = await sendAndRecv(ws, 4, "DOM.querySelector", {
    nodeId: rootNodeId,
    selector: "body",
  });
  const bodyNodeId = bodyResp.result.nodeId;

  // Get outerHTML
  const htmlResp = await sendAndRecv(ws, 5, "DOM.getOuterHTML", {
    nodeId: bodyNodeId,
  });
  return htmlResp.result.outerHTML;
}

/**
 * Convert HTML to markdown using defuddle.
 */
async function htmlToMarkdown(html, baseUrl) {
  const dom = new JSDOM(html, { url: baseUrl });
  const result = await Defuddle(dom.window.document, baseUrl, {
    markdown: true,
  });
  return result.content || "";
}

// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const args = {
    url: null,
    outputPath: null,
    wsUrl: null,
    renderTimeout: DEFAULT_RENDER_TIMEOUT,
  };

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "-w" || arg === "--ws-url") {
      args.wsUrl = argv[++i];
    } else if (arg === "-t" || arg === "--timeout") {
      args.renderTimeout = parseInt(argv[++i], 10);
    } else if (arg === "-h" || arg === "--help") {
      args.help = true;
    } else if (!args.url) {
      args.url = arg;
    } else if (!args.outputPath) {
      args.outputPath = arg;
    }
  }

  return args;
}

function printUsage() {
  console.log(`Usage: node scrape_page.mjs <url> [output.md] [options]
Options:
  -w, --ws-url <url>   CDP WebSocket URL (auto-detected by default)
  -t, --timeout <ms>  Wait time for JS rendering (default: ${DEFAULT_RENDER_TIMEOUT})
  -h, --help           Show this help message
Examples:
  node scrape_page.mjs "https://example.com"
  node scrape_page.mjs "https://example.com" "output.md"
  node tools/scrape_page.mjs "https://example.com" -t 5000
`);
}

async function main() {
  const args = parseArgs(process.argv);

  if (args.help || !args.url) {
    printUsage();
    process.exit(args.help ? 0 : 1);
    return;
  }

  // Determine CDP WebSocket URL
  let targets;
  let autoWsUrl;
  try {
    const resp = await fetch("http://localhost:9222/json");
    targets = await resp.json();
    autoWsUrl = findChromeDebuggerUrl(targets);
  } catch {
    console.error(
      "Error: Could not connect to Chrome debugger at http://localhost:9222/json"
    );
    console.error(
      "Is Chrome running with --remote-debugging-port=9222?"
    );
    process.exit(1);
    return;
  }

  const finalWsUrl = args.wsUrl || autoWsUrl;

  if (!finalWsUrl) {
    console.error(
      "Error: Could not find a Chrome debugger page."
    );
    process.exit(1);
    return;
  }

  console.error(`Connecting to Chrome at ${finalWsUrl}`);

  // Connect via WebSocket
  const ws = new WebSocket(finalWsUrl, {
    maxPayload: 200 * 1024 * 1024,
  });

  await new Promise((resolve, reject) => {
    ws.on("open", resolve);
    ws.on("error", reject);
  });

  const html = await navigateAndGetHtml(ws, args.url, args.renderTimeout);
  ws.close();

  // Save HTML to temp file (for potential debug) — only if we need it
  const htmlPath = join(tmpdir(), `scrape_${Date.now()}.html`);
  writeFileSync(htmlPath, html, "utf8");
  console.error(`HTML saved to: ${htmlPath}`);

  // Convert to markdown
  const markdown = await htmlToMarkdown(html, args.url);

  if (args.outputPath) {
    writeFileSync(args.outputPath, markdown, "utf8");
    console.error(`Markdown saved to: ${args.outputPath}`);
  } else {
    console.log(markdown);
  }

  // Cleanup temp HTML
  unlinkSync(htmlPath);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
