"""Pydantic models for API request/response."""
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4


class TranslationMode(str, Enum):
    """Translation mode enum."""
    FAST = "fast"
    NORMAL = "normal"
    FINE = "fine"


class ProjectStatus(str, Enum):
    """Project status enum."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    ERROR = "error"


class WorkflowStep(str, Enum):
    """Workflow step enum."""
    INIT = "init"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    TERMINOLOGY = "terminology"
    PROMPT_GEN = "prompt_gen"
    SEGMENT = "segment"
    TRANSLATE = "translate"
    REVIEW = "review"
    REVISE = "revise"
    POLISH = "polish"
    DONE = "done"


# Request Models
class CreateProjectRequest(BaseModel):
    """Request to create a new translation project."""
    url: str = Field(..., description="URL of the article to translate")
    mode: TranslationMode = Field(default=TranslationMode.NORMAL, description="Translation mode")
    target_language: str = Field(default="中文", description="Target language")


class ConfirmStepRequest(BaseModel):
    """Request to confirm a workflow step."""
    project_id: str = Field(..., description="Project ID")
    step: WorkflowStep = Field(..., description="Step to confirm")
    approved: bool = Field(default=True, description="Whether approved")
    modifications: Optional[str] = Field(default=None, description="User modifications to the content")


class UpdateTranslationRequest(BaseModel):
    """Request to update translation content."""
    project_id: str = Field(..., description="Project ID")
    file_path: str = Field(..., description="File path relative to project")
    content: str = Field(..., description="New content")


# Response Models
class ProjectMetadata(BaseModel):
    """Project metadata."""
    project_id: str = Field(default_factory=lambda: str(uuid4()))
    url: str
    title: Optional[str] = None
    mode: TranslationMode
    target_language: str
    status: ProjectStatus = ProjectStatus.PENDING
    current_step: WorkflowStep = WorkflowStep.INIT
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowState(BaseModel):
    """Workflow state for persistence."""
    project_id: str
    mode: TranslationMode
    current_step: WorkflowStep
    paused_at: Optional[str] = None
    step_results: Dict[str, Any] = Field(default_factory=dict)
    confirmation_queue: List[str] = Field(default_factory=list)


class ProjectDetail(ProjectMetadata):
    """Detailed project information."""
    workflow_state: Optional[WorkflowState] = None
    files: List[str] = Field(default_factory=list)


class ProjectListItem(BaseModel):
    """Project list item."""
    project_id: str
    url: str
    title: Optional[str]
    mode: TranslationMode
    status: ProjectStatus
    current_step: WorkflowStep
    created_at: datetime
    updated_at: datetime


class StepResult(BaseModel):
    """Result of a workflow step."""
    step: WorkflowStep
    status: str  # pending, completed, waiting_confirmation
    file_path: Optional[str] = None
    content: Optional[str] = None
    message: Optional[str] = None


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # progress, confirmation, error, complete
    project_id: str
    step: Optional[WorkflowStep] = None
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class LLMSettings(BaseModel):
    """LLM settings."""
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    api_base: Optional[str] = None  # Custom API URL