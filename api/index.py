"""
Vercel serverless function handler for FastAPI application.
This file acts as the entry point for all API routes on Vercel.
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variable for model directory
os.environ.setdefault("MODEL_DIR", str(project_root / "models"))

from app.main import app

# Vercel expects a handler function
def handler(request):
    """
    Vercel serverless function handler.
    
    Args:
        request: Vercel request object
        
    Returns:
        Response from FastAPI application
    """
    return app(request.environ, request.start_response)

# For Vercel Python runtime
if __name__ == "__main__":
    from mangum import Mangum
    handler = Mangum(app)

