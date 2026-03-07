"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.routers import translation, projects


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print(f"Starting Translation Agent API...")
    print(f"Projects directory: {settings.PROJECTS_DIR}")
    print(f"LLM Provider: {settings.LLM_PROVIDER}")

    # Ensure projects directory exists
    settings.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    print("Shutting down Translation Agent API...")


app = FastAPI(
    title="Translation Agent API",
    description="API for AI-powered translation with human-in-the-loop workflow",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(translation.router)
app.include_router(projects.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Translation Agent API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)