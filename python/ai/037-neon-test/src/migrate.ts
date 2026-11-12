import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import pg from "pg";
import { env } from "./env.js";

const { Client } = pg;

/** db/migrations, resolved relative to the compiled file (dist/src/migrate.js). */
const MIGRATIONS_DIR = fileURLToPath(new URL("../../db/migrations/", import.meta.url));

/**
 * Apply pending SQL migrations.
 *
 * Uses the direct (non-pooled) connection: PgBouncer's transaction mode does
 * not keep session state, so DDL run over the pooled connection is unsafe.
 */
export async function migrate(): Promise<string[]> {
  const client = new Client({ connectionString: env.databaseUrlUnpooled });
  await client.connect();
  const applied: string[] = [];

  try {
    await client.query(`CREATE TABLE IF NOT EXISTS schema_migrations (
      name       text PRIMARY KEY,
      applied_at timestamptz NOT NULL DEFAULT now()
    )`);

    const files = (await readdir(MIGRATIONS_DIR)).filter((f) => f.endsWith(".sql")).sort();

    for (const file of files) {
      const { rows } = await client.query("SELECT 1 FROM schema_migrations WHERE name = $1", [file]);
      if (rows.length > 0) {
        console.log(`skip    ${file} (already applied)`);
        continue;
      }

      const sql = await readFile(path.join(MIGRATIONS_DIR, file), "utf8");
      await client.query("BEGIN");
      try {
        await client.query(sql);
        await client.query("INSERT INTO schema_migrations (name) VALUES ($1)", [file]);
        await client.query("COMMIT");
        applied.push(file);
        console.log(`applied ${file}`);
      } catch (err) {
        await client.query("ROLLBACK");
        throw err;
      }
    }
  } finally {
    await client.end();
  }

  return applied;
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  migrate()
    .then((applied) => {
      console.log(applied.length === 0 ? "no pending migrations" : `applied ${applied.length} migration(s)`);
    })
    .catch((err) => {
      console.error("migration failed:", err instanceof Error ? err.message : err);
      process.exit(1);
    });
}
