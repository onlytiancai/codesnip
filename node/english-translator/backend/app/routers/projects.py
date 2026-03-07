"""Projects API routes."""
from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import ProjectListItem, ProjectDetail
from app.services.project_service import ProjectService

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=List[ProjectListItem])
async def list_projects():
    """List all translation projects."""
    project_service = ProjectService()
    return await project_service.list_projects()


@router.get("/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: str):
    """Get project details."""
    project_service = ProjectService()
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete a project."""
    project_service = ProjectService()
    success = await project_service.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted"}