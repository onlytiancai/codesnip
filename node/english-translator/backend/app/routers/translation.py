"""Translation API routes."""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import json

from app.models.schemas import (
    CreateProjectRequest,
    ConfirmStepRequest,
    UpdateTranslationRequest,
    ProjectDetail,
    ProjectListItem,
    StepResult,
    WebSocketMessage,
    ProjectStatus,
    WorkflowStep,
)
from app.services.project_service import ProjectService
from app.services.llm_service import LLMService
from app.workflows.translation_workflow import TranslationWorkflow

router = APIRouter(prefix="/api/translation", tags=["translation"])

# Store active WebSocket connections
active_connections: dict[str, List[WebSocket]] = {}


async def broadcast_progress(project_id: str, message: WebSocketMessage):
    """Broadcast progress to all connected WebSocket clients."""
    if project_id in active_connections:
        dead_connections = []
        for connection in active_connections[project_id]:
            try:
                await connection.send_json(message.model_dump())
            except Exception:
                dead_connections.append(connection)
        for dead in dead_connections:
            active_connections[project_id].remove(dead)


@router.post("/projects", response_model=ProjectDetail)
async def create_project(request: CreateProjectRequest):
    """Create a new translation project."""
    project_service = ProjectService()

    # Create project
    project = await project_service.create_project(
        url=request.url,
        mode=request.mode,
        target_language=request.target_language
    )

    return project


@router.get("/projects", response_model=List[ProjectListItem])
async def list_projects():
    """List all translation projects."""
    project_service = ProjectService()
    return await project_service.list_projects()


@router.get("/projects/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: str):
    """Get project details."""
    project_service = ProjectService()
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project."""
    project_service = ProjectService()
    success = await project_service.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted"}


@router.post("/projects/{project_id}/start")
async def start_translation(project_id: str):
    """Start the translation workflow."""
    project_service = ProjectService()
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status != ProjectStatus.PENDING:
        raise HTTPException(status_code=400, detail="Project already started")

    # Start workflow in background
    asyncio.create_task(run_workflow(project_id))

    return {"status": "started", "project_id": project_id}


@router.post("/projects/{project_id}/resume")
async def resume_translation(project_id: str):
    """Resume a paused translation workflow."""
    project_service = ProjectService()
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status != ProjectStatus.WAITING_CONFIRMATION:
        raise HTTPException(status_code=400, detail="Project not waiting for confirmation")

    # Resume workflow in background
    asyncio.create_task(run_workflow(project_id, resume=True))

    return {"status": "resumed", "project_id": project_id}


@router.post("/confirm", response_model=StepResult)
async def confirm_step(request: ConfirmStepRequest):
    """Confirm a workflow step."""
    project_service = ProjectService()
    project = await project_service.get_project(request.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = await project_service.confirm_step(
        project_id=request.project_id,
        step=request.step,
        approved=request.approved,
        modifications=request.modifications
    )

    return result


@router.post("/update", response_model=dict)
async def update_translation(request: UpdateTranslationRequest):
    """Update translation content."""
    project_service = ProjectService()
    success = await project_service.update_file(
        project_id=request.project_id,
        file_path=request.file_path,
        content=request.content
    )
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "updated"}


@router.get("/projects/{project_id}/files/{file_path:path}")
async def get_file(project_id: str, file_path: str):
    """Get a file from a project."""
    project_service = ProjectService()
    content = await project_service.get_file(project_id, file_path)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")
    return {"content": content}


@router.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()

    if project_id not in active_connections:
        active_connections[project_id] = []
    active_connections[project_id].append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            message = json.loads(data)
            # Echo back for now
            await websocket.send_json({"type": "ack", "message": "Received"})
    except WebSocketDisconnect:
        if project_id in active_connections:
            active_connections[project_id].remove(websocket)
    except Exception as e:
        if project_id in active_connections:
            active_connections[project_id].remove(websocket)


async def run_workflow(project_id: str, resume: bool = False):
    """Run the translation workflow."""
    project_service = ProjectService()
    workflow = TranslationWorkflow(project_service)

    async def progress_callback(step: WorkflowStep, message: str, data: dict = None):
        """Callback for workflow progress updates."""
        msg = WebSocketMessage(
            type="progress",
            project_id=project_id,
            step=step,
            message=message,
            data=data
        )
        await broadcast_progress(project_id, msg)

    async def confirmation_callback(step: WorkflowStep, content: str, file_path: str):
        """Callback for confirmation requests."""
        msg = WebSocketMessage(
            type="confirmation",
            project_id=project_id,
            step=step,
            data={
                "content": content,
                "file_path": file_path
            }
        )
        await broadcast_progress(project_id, msg)

    try:
        await workflow.run(
            project_id=project_id,
            resume=resume,
            progress_callback=progress_callback,
            confirmation_callback=confirmation_callback
        )
    except Exception as e:
        msg = WebSocketMessage(
            type="error",
            project_id=project_id,
            message=str(e)
        )
        await broadcast_progress(project_id, msg)