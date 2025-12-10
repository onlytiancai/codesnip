Project: Speech + IPA Demo (Vue3)

Purpose
- Small single-page demo that runs entirely in the browser. It tokenizes input text, looks up IPA using a local offline CSV file (no external API dependency), and uses the Web Speech API to read text aloud with per-word and per-sentence highlighting.

Quick facts for an AI agent
- Single HTML file: `index.html` contains the entire app (Vue UMD + inline script + Tailwind). Focus edits here.
- No build toolchain: app uses CDN JS (Vue, Tailwind) and runs as a static page. Changes to JS are edits to `index.html`.
-- Network dependency: None for IPA lookup — the app now uses `config/offlineIPA.csv` as the single source of IPA entries.