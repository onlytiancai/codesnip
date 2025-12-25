Project: Wawa English Speaker (Vue3)

Purpose
- Feature-rich English reader application that runs entirely in the browser. It tokenizes input text, looks up IPA using a local offline CSV file, provides word information, uses the Web Speech API for text-to-speech with per-word/sentence highlighting, and supports sentence translation via Ollama API.

Quick facts for an AI agent
- Modular Vue3 application: Uses component-based architecture with separate files for components and utilities.
- No build toolchain: App uses ES modules and runs as static pages. Simply open `index.html` in a browser.

Key directories and files:
- `index.html`: Main HTML structure and global styles
- `main.js`: Application entry point and core logic
- `components/`: Vue components
- `utils/`: Utility functions