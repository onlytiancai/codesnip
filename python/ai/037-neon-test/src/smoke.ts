import { closePool, pool } from "./db.js";
import { migrate } from "./migrate.js";
import { startServer } from "./server.js";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

/**
 * End-to-end proof that the app talks to Neon:
 * migrate -> boot the HTTP server -> exercise every route over HTTP -> clean up.
 */
async function main(): Promise<void> {
  console.log("→ applying migrations over the direct (unpooled) connection");
  await migrate();

  const { url, close } = await startServer(0);
  console.log(`→ server listening on ${url}`);
  const lines: string[] = [];

  try {
    const health = await fetch(`${url}/health`);
    const healthBody = (await health.json()) as { status: string; database: string; db: string };
    assert(health.status === 200, `GET /health returned ${health.status}`);
    assert(healthBody.database === "up", "GET /health did not report the database as up");
    lines.push(`GET  /health      → ${health.status} ${JSON.stringify(healthBody)}`);

    const created = await fetch(`${url}/api/notes`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ body: "hello from Neon" }),
    });
    const createdBody = (await created.json()) as { note: { id: number; body: string } };
    assert(created.status === 201, `POST /api/notes returned ${created.status}`);
    assert(typeof createdBody.note?.id === "number", "POST /api/notes did not return a note id");
    lines.push(`POST /api/notes  → ${created.status} ${JSON.stringify(createdBody.note)}`);

    const listed = await fetch(`${url}/api/notes`);
    const listedBody = (await listed.json()) as { notes: Array<{ id: number; body: string }> };
    assert(listed.status === 200, `GET /api/notes returned ${listed.status}`);
    assert(
      listedBody.notes.some((n) => n.id === createdBody.note.id),
      "GET /api/notes did not return the note created a moment ago",
    );
    lines.push(`GET  /api/notes  → ${listed.status} ${listedBody.notes.length} note(s) in the table`);

    const invalid = await fetch(`${url}/api/notes`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ body: "" }),
    });
    assert(invalid.status === 400, `POST /api/notes with empty body returned ${invalid.status}`);
    lines.push(`POST /api/notes  → ${invalid.status} (empty body rejected)`);

    const missing = await fetch(`${url}/nope`);
    assert(missing.status === 404, `GET /nope returned ${missing.status}`);
    lines.push(`GET  /nope        → ${missing.status} (unknown route)`);

    const deleted = await pool.query("DELETE FROM notes WHERE id = $1", [createdBody.note.id]);
    lines.push(`cleanup          → deleted ${deleted.rowCount ?? 0} test row(s)`);
  } finally {
    await close();
    await closePool();
  }

  console.log(lines.map((l) => `  ${l}`).join("\n"));
  console.log("SMOKE TEST PASSED");
}

main().catch((err: unknown) => {
  console.error("SMOKE TEST FAILED:", err instanceof Error ? err.message : err);
  process.exit(1);
});
