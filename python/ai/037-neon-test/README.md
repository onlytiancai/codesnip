# neon-notes

A minimal notes API backed by **Neon (Lakebase) Postgres**.

- Neon project: `proud-morning-83814385` (org `org-sparkling-grass-37002835`, branch `production`)
- Driver: `pg` (node-postgres) over the **pooled** endpoint for app traffic, **direct** endpoint for migrations
- Env vars: typed and validated with `@neon/env`'s `parseEnv`, derived from `neon.ts`

## Layout

| Path | Purpose |
| --- | --- |
| `neon.ts` | Neon infrastructure-as-code (services declared per branch) |
| `.neon` | Local link: org / project / branch (git-ignored) |
| `.env` | Neon-managed env vars pulled by `neon deploy` (git-ignored, contains secrets) |
| `.mcp.json` | Project-level Neon MCP server (OAuth — no key in the file) |
| `.claude/skills/` | Neon agent skills (`neon`, `neon-postgres`) installed by `neon skills` |
| `db/migrations/` | SQL migrations, applied in order and recorded in `schema_migrations` |
| `src/env.ts` | Typed env (`parseEnv`) |
| `src/db.ts` | `pg` pool on the pooled connection string |
| `src/migrate.ts` | Migration runner (direct connection) |
| `src/server.ts` | HTTP API |
| `src/smoke.ts` | End-to-end proof (migrate → boot → HTTP → DB) |

## API

| Method | Path | Result |
| --- | --- | --- |
| `GET` | `/health` | `{ status, database, db, now }` — runs `SELECT current_database(), now()` |
| `GET` | `/api/notes` | `{ notes: [...] }` |
| `POST` | `/api/notes` | `{ note }` — body `{ "body": "..." }`, `400` on empty body |

## Commands

```bash
pnpm install
pnpm build      # tsc -p tsconfig.json  (emits dist/)
pnpm migrate    # applies db/migrations/*.sql over the direct connection
pnpm start      # node --env-file-if-exists=.env dist/src/server.js  (PORT=3000)
pnpm smoke      # end-to-end check against the live Neon database
```

`npm run build` / `npm start` / `npm run smoke` work identically.

## Neon CLI

```bash
neon status                      # live config of the linked branch
neon config plan                 # dry-run diff of what deploy would change
neon deploy                      # apply neon.ts + pull env into .env
neon checkout dev --create       # branch-first dev flow, pulls that branch's env
neon skills status               # are the installed Neon skills current?
```

## Notes

- `neon deploy` pulls `DATABASE_URL` (pooled), `DATABASE_URL_UNPOOLED` (direct) and `NEON_BRANCH`.
- App traffic uses the pooled string; migrations use the direct string — PgBouncer's
  transaction mode does not preserve session state, so DDL must not run through the pool.
- Neon agent skills are installed per project; `neon skills update -y` refreshes them.
