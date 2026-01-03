"""API endpoints for transaction prediction."""
from fastapi import APIRouter, HTTPException, Depends
import pandas as pd
import logging
from typing import Optional
from app.schemas.transaction import TransactionRequest, PredictionResponse
from app.services.model_service import ModelService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton model service instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """
    Get or create the singleton model service instance.
    
    Returns:
        ModelService: The initialized model service instance
    """
    global _model_service
    if _model_service is None:
        _model_service = ModelService(settings.MODEL_DIR)
    return _model_service


def calculate_risk_level(score: float) -> str:
    """
    Calculate risk level based on score.
    
    Args:
        score: Risk score between 0 and 1
        
    Returns:
        Risk level string: "LOW", "HIGH", or "CRITICAL"
    """
    if score > 0.8:
        return "CRITICAL"
    elif score > 0.5:
        return "HIGH"
    else:
        return "LOW"


@router.post("/predict", response_model=PredictionResponse, summary="Predict transaction risk")
def predict_transaction(
    txn: TransactionRequest,
    service: ModelService = Depends(get_model_service)
) -> PredictionResponse:
    """
    Predict the risk of money laundering for a transaction.
    
    Args:
        txn: Transaction data to analyze
        service: Model service dependency
        
    Returns:
        PredictionResponse with risk score and level
        
    Raises:
        HTTPException: 400 for invalid input, 503 for service unavailable, 500 for server errors
    """
    try:
        # Convert Pydantic model to DataFrame (by_alias handles "From Bank" vs "from_bank")
        data = txn.model_dump(by_alias=True)
        df = pd.DataFrame([data])

        # Get prediction from service
        score = service.predict(df)

        return PredictionResponse(
            is_laundering=int(score > 0.5),
            risk_score=round(score, 4),
            risk_level=calculate_risk_level(score)
        )
    except RuntimeError as e:
        # Model not loaded
        logger.error(f"Model service error: {e}")
        raise HTTPException(status_code=503, detail="Model service unavailable")
    except ValueError as e:
        # Invalid input data
        logger.warning(f"Invalid input data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # General server error
        logger.exception(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
