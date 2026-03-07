"""Workflow state definition."""
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"


class TranslationState(BaseModel):
    """State for translation workflow."""

    # Project info
    project_id: str
    url: str
    mode: str = "normal"  # fast, normal, fine
    target_language: str = "中文"

    # Content
    original_content: str = ""
    title: Optional[str] = None

    # Workflow progress
    current_step: str = "init"
    step_statuses: Dict[str, StepStatus] = Field(default_factory=dict)

    # Step outputs
    analysis: str = ""
    terminology: str = ""
    translation_prompt: str = ""
    segments: List[str] = Field(default_factory=list)
    translations: List[str] = Field(default_factory=list)
    draft: str = ""
    critique: str = ""
    revision: str = ""
    final_translation: str = ""

    # Confirmation tracking
    confirmations_needed: List[str] = Field(default_factory=list)
    confirmations_received: Dict[str, bool] = Field(default_factory=dict)

    # File paths for outputs
    file_paths: Dict[str, str] = Field(default_factory=dict)

    # Metadata
    total_chunks: int = 0
    completed_chunks: int = 0

    class Config:
        use_enum_values = True


# Confirmation points for different modes
FAST_MODE_STEPS = ["extract", "translate", "done"]

NORMAL_MODE_STEPS = [
    "extract",
    "analyze",
    "terminology",
    "prompt_gen",
    "segment",
    "translate",
    "done"
]

FINE_MODE_STEPS = [
    "extract",
    "analyze",
    "terminology",
    "prompt_gen",
    "segment",
    "translate",
    "review",
    "revise",
    "polish",
    "done"
]

# Steps that require user confirmation
CONFIRMATION_STEPS = {
    "normal": ["analyze", "prompt_gen"],
    "fine": ["analyze", "prompt_gen", "review"]
}