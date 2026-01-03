"""Main FastAPI application."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.v1.endpoints import router as api_router, get_model_service
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    try:
        logger.info("Initializing model service...")
        get_model_service()  # This will load the model
        logger.info("Application startup complete.")
    except Exception as e:
        logger.error(f"Failed to initialize model service: {e}")
        raise
    
    yield
    
    # Shutdown (if needed in future)
    logger.info("Application shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Anti-Money Laundering detection system using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

# Mount Frontend (only if directory exists, for Vercel we serve from public/)
import os
if os.path.exists("app/static"):
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", summary="Root endpoint", tags=["Frontend"])
def read_root() -> FileResponse:
    """
    Serve the frontend HTML page.
    
    Returns:
        FileResponse: The frontend index.html file
    """
    return FileResponse('app/static/index.html')


@app.get("/health", summary="Health check", tags=["Health"])
def health_check() -> dict:
    """
    Health check endpoint to verify API is running.
    
    Returns:
        dict: Status information
    """
    try:
        # Verify model service is available
        service = get_model_service()
        model_status = "ready" if service.model is not None else "not_ready"
        
        return {
            "status": "ok",
            "model_status": model_status,
            "service": "AML Detection System"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "model_status": "error",
            "error": str(e)
        }
