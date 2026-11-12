CREATE TABLE IF NOT EXISTS notes (
  id         serial PRIMARY KEY,
  body       text        NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
