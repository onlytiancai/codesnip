import { parseEnv } from "@neon/env";
import config from "../neon.js";

/**
 * Typed, validated environment derived from the neon.ts config.
 *
 * `parseEnv` reads process.env (loaded from .env by `node --env-file-if-exists=.env`)
 * and validates exactly the variables we ask for, so a missing or malformed
 * DATABASE_URL fails loudly at startup instead of at the first query.
 */
const { postgres } = parseEnv(config, ["DATABASE_URL", "DATABASE_URL_UNPOOLED"]);

export const env = {
  /** Pooled connection (PgBouncer, transaction mode) — application traffic. */
  databaseUrl: postgres.databaseUrl,
  /** Direct connection — migrations and session-scoped work. */
  databaseUrlUnpooled: postgres.databaseUrlUnpooled,
};
