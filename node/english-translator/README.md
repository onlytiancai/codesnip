# Translation Agent

AI-powered translation application with human-in-the-loop workflow, built with FastAPI + LangChain backend and Vue 3 + Tailwind CSS frontend.

## Features

- **Three Translation Modes**:
  - **Fast**: Quick translation without analysis
  - **Normal**: Analysis, terminology extraction, and translation (default)
  - **Fine**: Full workflow with review, revision, and polish

- **Human-in-the-loop**: Key steps require user confirmation before proceeding

- **Project Management**: Each translation creates a project directory with all intermediate files

- **WebSocket Real-time Updates**: Live progress updates during translation

- **Preview & Edit**: Side-by-side view of original and translation with editing capability

- **Custom API Support**: Use custom LLM endpoints (Azure OpenAI, local LLMs, proxies)

## Project Structure

```
english-translator/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # FastAPI application entry
│   │   ├── config.py          # Configuration
│   │   ├── routers/           # API routes
│   │   ├── services/          # Business logic services
│   │   ├── workflows/         # LangChain translation workflow
│   │   └── models/            # Pydantic models
│   ├── projects/              # Project storage
│   └── requirements.txt
├── frontend/                  # Vue 3 frontend
│   ├── src/
│   │   ├── components/        # Vue components
│   │   ├── views/             # Page views
│   │   ├── stores/            # Pinia state management
│   │   └── api/               # API client
│   └── package.json
└── docs/
    └── baoyu-fanyi.md         # Reference documentation
```

## Setup

### Backend Setup

1. Create a virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

4. Run the server:
```bash
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
pnpm install
```

2. Run the development server:
```bash
pnpm dev
```

3. Open http://localhost:5173 in your browser

## API Endpoints

### Projects

- `GET /api/projects` - List all projects
- `GET /api/projects/{id}` - Get project details
- `POST /api/translation/projects` - Create new project
- `DELETE /api/projects/{id}` - Delete project

### Translation

- `POST /api/translation/projects/{id}/start` - Start translation
- `POST /api/translation/projects/{id}/resume` - Resume translation
- `POST /api/translation/confirm` - Confirm workflow step
- `POST /api/translation/update` - Update translation content
- `GET /api/translation/projects/{id}/files/{path}` - Get file content

### WebSocket

- `WS /ws/api/translation/{id}` - Real-time updates

## Translation Workflow

### Normal Mode (Default)

1. **Extract** - Fetch and convert URL content to Markdown
2. **Analyze** - Analyze content, extract terminology, identify challenges
3. **Terminology** - Build terminology table
4. **Generate Prompt** - Create translation prompt
5. **Segment** - Split long content into chunks
6. **Translate** - Parallel translation of chunks
7. **Complete** - Finalize translation

### Fine Mode

Extends Normal mode with:
8. **Review** - Critique translation quality
9. **Revise** - Apply review suggestions
10. **Polish** - Final polish for flow and style

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai/anthropic) | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_MODEL` | OpenAI model | `gpt-4o` |
| `ANTHROPIC_MODEL` | Anthropic model | `claude-sonnet-4-6` |
| `OPENAI_API_BASE` | Custom OpenAI API URL | - |
| `ANTHROPIC_API_BASE` | Custom Anthropic API URL | - |
| `PROJECTS_DIR` | Projects storage directory | `./projects` |
| `DEFAULT_TRANSLATION_MODE` | Default mode | `normal` |
| `DEFAULT_TARGET_LANGUAGE` | Default target language | `中文` |

### Using Custom API Endpoints

You can use custom LLM endpoints by setting the `OPENAI_API_BASE` or `ANTHROPIC_API_BASE` environment variables:

```env
# Example: Using a local LLM server
OPENAI_API_BASE=http://localhost:8000/v1

# Example: Using Azure OpenAI
OPENAI_API_BASE=https://your-resource.openai.azure.com/openai/deployments/your-deployment
```

## Output Files

Each project saves these files:

```
projects/{project_id}/
├── metadata.json           # Project metadata
├── workflow_state.json     # Workflow state
├── original.md             # Original content
├── 01-analysis.md          # Content analysis
├── 02-terminology.md       # Terminology table
├── 03-prompt.md            # Translation prompt
├── 04-segments/            # Content segments
├── 05-translations/        # Segment translations
├── 06-draft.md             # Merged draft
├── 07-critique.md          # Review feedback
├── 08-revision.md          # Revised version
└── translation.md          # Final translation
```

## License

MIT