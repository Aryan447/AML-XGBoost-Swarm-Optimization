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

# Set environment variable for model directory (Vercel uses /var/task for serverless functions)
model_dir = os.getenv("MODEL_DIR", str(project_root / "models"))
os.environ["MODEL_DIR"] = model_dir

from mangum import Mangum
from app.main import app

# Create Mangum adapter for Vercel
handler = Mangum(app, lifespan="off")

