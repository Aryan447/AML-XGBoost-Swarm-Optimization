"""Application configuration settings."""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Attributes:
        PROJECT_NAME: Name of the project
        API_V1_STR: API version prefix
        MODEL_DIR: Directory containing model artifacts
    """
    
    PROJECT_NAME: str = "AML Detection System"
    API_V1_STR: str = "/api/v1"
    # Support both local paths and URLs for model loading
    # For Vercel: Set MODEL_URL environment variable to host models externally
    MODEL_DIR: str = os.getenv(
        "MODEL_DIR", 
        os.getenv(
            "MODEL_URL", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        )
    )

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
