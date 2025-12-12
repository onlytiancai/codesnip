Project: Wawa English Speaker (Vue3)

Purpose
- Feature-rich English reader application that runs entirely in the browser. It tokenizes input text, looks up IPA using a local offline CSV file, provides word information, uses the Web Speech API for text-to-speech with per-word/sentence highlighting, and supports sentence translation via Ollama API.

Quick facts for an AI agent
- 不要尝试用 python3 -m http.server 8000 来测试页面，直接跳过测试
- Modular Vue3 application: Uses component-based architecture with separate files for components and utilities.
- Main entry point: `main.js` handles application setup and logic.
- Components directory: `components/` contains Vue components (Sentence.js, WordBlock.js).
- Utilities directory: `utils/` contains modular helper functions:
  - `ipa.js`: Offline IPA lookup from CSV file
  - `tokenizer.js`: Text tokenization and analysis
  - `speech.js`: Text-to-speech functionality
  - `dictionary.js`: Word information lookup
  - `translation.js`: Ollama API translation integration
- No build toolchain: App uses ES modules and runs as static pages. Simply open `index.html` in a browser.
- Network dependencies:
  - None for IPA lookup: Uses `config/offlineIPA.csv` as the single source of IPA entries
  - Optional for translation: Requires Ollama service running locally for sentence translation
- Settings persistence: User settings saved to localStorage

Key directories and files:
- `index.html`: Main HTML structure and global styles
- `main.js`: Application entry point and core logic
- `components/`: Vue components
- `utils/`: Utility functions
- `config/`: Configuration files (offlineIPA.csv)
- `lib/`: External libraries (Vue3)