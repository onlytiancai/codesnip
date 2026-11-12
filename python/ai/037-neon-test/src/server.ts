import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { pathToFileURL } from "node:url";
import { pool } from "./db.js";

export type Note = {
  id: number;
  body: string;
  created_at: string;
};

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    size += buf.length;
    if (size > 64 * 1024) throw new Error("request body too large");
    chunks.push(buf);
  }
  if (chunks.length === 0) return undefined;
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://localhost");

  if (req.method === "GET" && url.pathname === "/health") {
    const { rows } = await pool.query<{ db: string; now: Date }>(
      "SELECT current_database() AS db, now() AS now",
    );
    return sendJson(res, 200, { status: "ok", database: "up", ...rows[0] });
  }

  if (req.method === "GET" && url.pathname === "/api/notes") {
    const { rows } = await pool.query<Note>("SELECT id, body, created_at FROM notes ORDER BY id");
    return sendJson(res, 200, { notes: rows });
  }

  if (req.method === "POST" && url.pathname === "/api/notes") {
    const payload = await readJsonBody(req);
    const body =
      typeof payload === "object" && payload !== null ? (payload as { body?: unknown }).body : undefined;
    if (typeof body !== "string" || body.trim() === "") {
      return sendJson(res, 400, { error: "body must be a non-empty string" });
    }
    const { rows } = await pool.query<Note>(
      "INSERT INTO notes (body) VALUES ($1) RETURNING id, body, created_at",
      [body],
    );
    return sendJson(res, 201, { note: rows[0] });
  }

  return sendJson(res, 404, { error: "not_found" });
}

export function createApp(): Server {
  return createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      console.error("[http]", err instanceof Error ? err.message : err);
      if (!res.headersSent) sendJson(res, 500, { error: "internal_error" });
      else res.end();
    });
  });
}

export function startServer(port = Number(process.env.PORT ?? 3000)): Promise<{
  url: string;
  close: () => Promise<void>;
}> {
  const server = createApp();
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, () => {
      const address = server.address();
      const actualPort = typeof address === "object" && address !== null ? address.port : port;
      resolve({
        url: `http://127.0.0.1:${actualPort}`,
        close: () => new Promise<void>((done) => server.close(() => done())),
      });
    });
  });
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  const { url } = await startServer();
  console.log(`notes API listening on ${url}`);
}
