Project: Speech + IPA Demo (Vue3)

Purpose
- Small single-page demo that runs entirely in the browser. It tokenizes input text, looks up IPA using a local offline CSV file (no external API dependency), caches results in localStorage, and uses the Web Speech API to read text aloud with per-word and per-sentence highlighting.

Quick facts for an AI agent
- Single HTML file: `index.html` contains the entire app (Vue UMD + inline script + Tailwind). Focus edits here.
- No build toolchain: app uses CDN JS (Vue, Tailwind) and runs as a static page. Changes to JS are edits to `index.html`.
-- Network dependency: None for IPA lookup — the app now uses `config/offlineIPA.csv` as the single source of IPA entries. localStorage caching (`ipa_cache_v1`) is still used to store resolved IPA values.

Key files / patterns
- `index.html` — app logic (setup(), tokenizePreserve, analyze, speak, speakSentences, speakWord). Most PRs will touch this file.
- `README.md` — short project summary; use it as a doc source for intent.

Architecture & invariants
- The app is a single Vue 3 composition-API component mounted to `#app`.
- Word tokens are stored in a reactive `wordBlocks` array; each item: { word, ipa, highlight, sentenceIndex }.
- Sentences are determined by split tokens (.,!? and Chinese punctuation); tests or changes that affect sentence-splitting must update both `tokenizePreserve` and the logic in `analyze` that assigns `sentenceIndex`.
- Speech behavior uses the browser `speechSynthesis` APIs and `SpeechSynthesisUtterance`. Code paths must guard against `speechSynthesis` being undefined (project already includes such guards).

Developer workflows
- Run locally by opening `index.html` directly in a browser, or (recommended) use a local development server such as the "Live Server" VS Code extension which handles file serving and avoids file:// fetch restrictions. The project no longer requires `python3 -m http.server`.
- No npm install or build required for typical edits. If adding new dependencies or tooling, include a README section and a package manifest (`package.json`).

Project-specific conventions
- Single-file app: prefer editing `index.html` instead of splitting into many files unless you add a minimal build setup and explain it in README.
- Keep the offline IPA map (`offlineIPA`) and `CACHE_KEY` semantics consistent. When adding new IPA providers or changing cache shape, keep backward compatibility with existing cache entries.
- Tokenization must preserve original punctuation and whitespace surrounding tokens for correct visual rendering; use `tokenizePreserve` as canonical implementation.

Testing and safety
- Manual smoke test: open page in a browser without network access to verify offline fallback IPA mapping and that UI shows friendly message if `speechSynthesis` is unavailable.
- When modifying speech flows (utterance lifecycle handlers), ensure highlight clearing on `onend`/`onerror` to avoid stale UI state.

When editing
- Prefer small, focused changes to `index.html`. If you must add files (tests, scripts), update `README.md` with run steps.
- Include examples when changing tokenization or IPA resolution (e.g., sample input string and expected token/ipa pairs).

Common edit examples to guide the agent
- Add support detection banner: update `setup()` and template to expose `speechSupported` and conditionally render a notice.
- Make speak buttons disabled when not supported: bind `:disabled="!speechSupported"` on buttons and add an explanatory `title`.
- Add new IPA source: add a `fetchIPA` branch that queries the service and falls back to `offlineIPA`; update `CACHE_KEY` value if cache schema changes.

If unsure
- Read `index.html` top-to-bottom. The app has all behavior in that single file; changes elsewhere are rare.
- Ask for clarification if a change touches cross-cutting concerns (speech timing, tokenization, caching) — these affect UX and are easy to regress.

Please confirm if you want a dismissible banner or persistence for the speech-support hint; I can implement and add test instructions.
