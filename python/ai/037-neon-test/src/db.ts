import pg from "pg";
import { env } from "./env.js";

const { Pool } = pg;

/**
 * node-postgres over Neon's pooled endpoint.
 *
 * The app is a long-running Node server, so a process-local pool is the right
 * shape: PgBouncer (the `-pooler` hostname) multiplexes many client connections
 * onto the Neon compute.
 */
export const pool = new Pool({
  connectionString: env.databaseUrl,
  max: 5,
  idleTimeoutMillis: 10_000,
  connectionTimeoutMillis: 15_000,
});

pool.on("error", (err) => {
  console.error("[db] idle client error:", err.message);
});

export async function closePool(): Promise<void> {
  await pool.end();
}
