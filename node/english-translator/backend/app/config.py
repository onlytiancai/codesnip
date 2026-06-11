"""Application configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""

    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # Custom API URLs (optional, for using custom endpoints)
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "")  # e.g., https://api.openai.com/v1
    ANTHROPIC_API_BASE: str = os.getenv("ANTHROPIC_API_BASE", "")  # e.g., https://api.anthropic.com

    # Application Settings
    PROJECTS_DIR: Path = Path(os.getenv("PROJECTS_DIR", "./projects"))
    DEFAULT_TRANSLATION_MODE: str = os.getenv("DEFAULT_TRANSLATION_MODE", "normal")
    DEFAULT_TARGET_LANGUAGE: str = os.getenv("DEFAULT_TARGET_LANGUAGE", "中文")

    # Translation Settings
    CHUNK_SIZE_THRESHOLD: int = 3000  # Characters threshold for chunking
    MAX_CHUNK_SIZE: int = 2000  # Maximum characters per chunk

    def __post_init__(self):
        """Ensure projects directory exists."""
        self.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()