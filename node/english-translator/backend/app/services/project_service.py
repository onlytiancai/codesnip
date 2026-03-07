"""Project management service."""
import json
import aiofiles
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4

from app.config import settings
from app.models.schemas import (
    ProjectMetadata,
    ProjectDetail,
    ProjectListItem,
    WorkflowState,
    WorkflowStep,
    ProjectStatus,
    TranslationMode,
    StepResult,
)


class ProjectService:
    """Service for managing translation projects."""

    def __init__(self):
        self.projects_dir = settings.PROJECTS_DIR
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def _get_project_dir(self, project_id: str) -> Path:
        """Get project directory path."""
        return self.projects_dir / project_id

    def _get_metadata_path(self, project_id: str) -> Path:
        """Get metadata file path."""
        return self._get_project_dir(project_id) / "metadata.json"

    def _get_workflow_state_path(self, project_id: str) -> Path:
        """Get workflow state file path."""
        return self._get_project_dir(project_id) / "workflow_state.json"

    async def create_project(
        self,
        url: str,
        mode: TranslationMode,
        target_language: str
    ) -> ProjectDetail:
        """Create a new translation project."""
        project_id = str(uuid4())[:8]
        project_dir = self._get_project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.utcnow()
        metadata = ProjectMetadata(
            project_id=project_id,
            url=url,
            mode=mode,
            target_language=target_language,
            status=ProjectStatus.PENDING,
            current_step=WorkflowStep.INIT,
            created_at=now,
            updated_at=now
        )

        # Save metadata
        metadata_path = self._get_metadata_path(project_id)
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(metadata.model_dump_json(indent=2))

        # Initialize workflow state
        workflow_state = WorkflowState(
            project_id=project_id,
            mode=mode,
            current_step=WorkflowStep.INIT,
            step_results={}
        )
        await self._save_workflow_state(project_id, workflow_state)

        return ProjectDetail(
            **metadata.model_dump(),
            workflow_state=workflow_state,
            files=[]
        )

    async def get_project(self, project_id: str) -> Optional[ProjectDetail]:
        """Get project details."""
        metadata = await self._load_metadata(project_id)
        if not metadata:
            return None

        workflow_state = await self._load_workflow_state(project_id)
        files = await self._list_project_files(project_id)

        return ProjectDetail(
            **metadata.model_dump(),
            workflow_state=workflow_state,
            files=files
        )

    async def list_projects(self) -> List[ProjectListItem]:
        """List all projects."""
        projects = []
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_dir():
                metadata = await self._load_metadata(project_dir.name)
                if metadata:
                    projects.append(ProjectListItem(**metadata.model_dump()))

        # Sort by updated_at descending
        projects.sort(key=lambda p: p.updated_at, reverse=True)
        return projects

    async def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        import shutil
        project_dir = self._get_project_dir(project_id)
        if not project_dir.exists():
            return False
        shutil.rmtree(project_dir)
        return True

    async def update_project_status(
        self,
        project_id: str,
        status: ProjectStatus,
        current_step: Optional[WorkflowStep] = None
    ) -> Optional[ProjectMetadata]:
        """Update project status."""
        metadata = await self._load_metadata(project_id)
        if not metadata:
            return None

        metadata.status = status
        if current_step:
            metadata.current_step = current_step
        metadata.updated_at = datetime.utcnow()

        await self._save_metadata(project_id, metadata)
        return metadata

    async def update_project_title(self, project_id: str, title: str):
        """Update project title."""
        metadata = await self._load_metadata(project_id)
        if metadata:
            metadata.title = title
            metadata.updated_at = datetime.utcnow()
            await self._save_metadata(project_id, metadata)

    async def save_file(
        self,
        project_id: str,
        filename: str,
        content: str
    ) -> bool:
        """Save a file to the project."""
        project_dir = self._get_project_dir(project_id)
        if not project_dir.exists():
            return False

        file_path = project_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)

        return True

    async def get_file(self, project_id: str, file_path: str) -> Optional[str]:
        """Get file content."""
        full_path = self._get_project_dir(project_id) / file_path
        if not full_path.exists():
            return None

        async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
            return await f.read()

    async def update_file(
        self,
        project_id: str,
        file_path: str,
        content: str
    ) -> bool:
        """Update file content."""
        return await self.save_file(project_id, file_path, content)

    async def update_workflow_state(
        self,
        project_id: str,
        state: WorkflowState
    ) -> bool:
        """Update workflow state."""
        return await self._save_workflow_state(project_id, state)

    async def confirm_step(
        self,
        project_id: str,
        step: WorkflowStep,
        approved: bool,
        modifications: Optional[str] = None
    ) -> StepResult:
        """Confirm a workflow step."""
        state = await self._load_workflow_state(project_id)
        if not state:
            raise ValueError("Project not found")

        # Store confirmation
        state.step_results[f"{step}_confirmed"] = approved
        if modifications:
            state.step_results[f"{step}_modifications"] = modifications

        await self._save_workflow_state(project_id, state)

        return StepResult(
            step=step,
            status="confirmed" if approved else "rejected",
            message=f"Step {step} {'approved' if approved else 'rejected'}"
        )

    async def get_workflow_state(self, project_id: str) -> Optional[WorkflowState]:
        """Get workflow state."""
        return await self._load_workflow_state(project_id)

    async def _load_metadata(self, project_id: str) -> Optional[ProjectMetadata]:
        """Load project metadata."""
        metadata_path = self._get_metadata_path(project_id)
        if not metadata_path.exists():
            return None

        async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return ProjectMetadata.model_validate_json(content)

    async def _save_metadata(self, project_id: str, metadata: ProjectMetadata):
        """Save project metadata."""
        metadata_path = self._get_metadata_path(project_id)
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(metadata.model_dump_json(indent=2))

    async def _load_workflow_state(self, project_id: str) -> Optional[WorkflowState]:
        """Load workflow state."""
        state_path = self._get_workflow_state_path(project_id)
        if not state_path.exists():
            return None

        async with aiofiles.open(state_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return WorkflowState.model_validate_json(content)

    async def _save_workflow_state(self, project_id: str, state: WorkflowState) -> bool:
        """Save workflow state."""
        state_path = self._get_workflow_state_path(project_id)
        async with aiofiles.open(state_path, 'w', encoding='utf-8') as f:
            await f.write(state.model_dump_json(indent=2))
        return True

    async def _list_project_files(self, project_id: str) -> List[str]:
        """List all files in a project."""
        project_dir = self._get_project_dir(project_id)
        if not project_dir.exists():
            return []

        files = []
        for path in project_dir.rglob('*'):
            if path.is_file() and path.name not in ['metadata.json', 'workflow_state.json']:
                files.append(str(path.relative_to(project_dir)))

        return sorted(files)