"""Model service for loading and using the trained XGBoost model."""
import os
import logging
import tempfile
from typing import Optional
from urllib.request import urlretrieve
import joblib
import xgboost as xgb
import pandas as pd

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing and using the trained AML detection model."""
    
    def __init__(self, model_dir: str) -> None:
        """
        Initialize the model service and load artifacts.
        
        Args:
            model_dir: Directory path containing model artifacts
            
        Raises:
            FileNotFoundError: If required model files are missing
            ValueError: If model files are corrupted or invalid
        """
        self.model: Optional[xgb.Booster] = None
        self.scaler: Optional[object] = None
        self.features: Optional[list] = None
        self._load_artifacts(model_dir)

    def _load_artifacts(self, model_dir: str) -> None:
        """
        Load model artifacts from the specified directory or URL.
        
        Supports:
        - Local file paths
        - URLs (downloads to temp directory)
        - Environment variables for model URLs
        
        Args:
            model_dir: Directory path or base URL containing model artifacts
            
        Raises:
            FileNotFoundError: If any required file is missing
            ValueError: If files are corrupted or invalid
        """
        try:
            # Check if model_dir is a URL or local path
            is_url = model_dir.startswith(('http://', 'https://'))
            
            if is_url:
                logger.info(f"Loading models from URL: {model_dir}...")
                # Download models to temp directory
                temp_dir = tempfile.mkdtemp()
                model_path = os.path.join(temp_dir, "best_model_gwo.json")
                scaler_path = os.path.join(temp_dir, "scaler.pkl")
                features_path = os.path.join(temp_dir, "feature_columns.pkl")
                
                # Download files
                logger.info("Downloading model files...")
                urlretrieve(f"{model_dir}/best_model_gwo.json", model_path)
                urlretrieve(f"{model_dir}/scaler.pkl", scaler_path)
                urlretrieve(f"{model_dir}/feature_columns.pkl", features_path)
                logger.info("Model files downloaded successfully")
            else:
                logger.info(f"Loading models from local path: {model_dir}...")
                model_path = os.path.join(model_dir, "best_model_gwo.json")
                scaler_path = os.path.join(model_dir, "scaler.pkl")
                features_path = os.path.join(model_dir, "feature_columns.pkl")

            # Verify files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Features not found at {features_path}")

            # Load artifacts
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.features = joblib.load(features_path)
            
            if not isinstance(self.features, list) or len(self.features) == 0:
                raise ValueError("Invalid feature columns file")
                
            logger.info(f"Models loaded successfully. Features: {len(self.features)}")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading models: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading models: {e}")
            raise ValueError(f"Failed to load model artifacts: {e}") from e

    def predict(self, input_df: pd.DataFrame) -> float:
        """
        Predict the risk score for a transaction.
        
        Args:
            input_df: DataFrame containing transaction data
            
        Returns:
            Risk score between 0 and 1
            
        Raises:
            RuntimeError: If model is not initialized
            ValueError: If input data is invalid
        """
        if not self.model or not self.scaler or not self.features:
            raise RuntimeError("Model service is not properly initialized.")

        if input_df.empty:
            raise ValueError("Input DataFrame is empty")

        # Create a copy to avoid modifying the original
        df = input_df.copy()

        # --- 1. Preprocess Timestamp ---
        if 'Timestamp' in df.columns:
            try:
                # Try converting to datetime -> int
                df['Timestamp'] = pd.to_datetime(df['Timestamp']).astype(int) / 10**9
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse timestamp: {e}. Setting to 0.")
                df['Timestamp'] = 0.0

        # --- 2. Preprocess Categorical Text ---
        # Map known values (Payment Format)
        pay_map = {
            'Cash': 1, 'Cheque': 2, 'ACH': 3, 'Credit Card': 4,
            'Wire': 5, 'Bitcoin': 6, 'Reinvestment': 7
        }
        if 'Payment Format' in df.columns:
            df['Payment Format'] = df['Payment Format'].map(pay_map).fillna(0)

        # Handle text fields that don't have LabelEncoders from training
        # Set to 0 to prevent type conversion errors
        text_cols = ['Account', 'Account.1', 'Receiving Currency', 'Payment Currency']
        for col in text_cols:
            if col in df.columns:
                df[col] = 0

        # --- 3. Align Columns & Force Numeric ---
        df_ready = pd.DataFrame(columns=self.features)

        for col in self.features:
            if col in df.columns:
                val = df[col]
            else:
                val = 0

            # Ensure data is numeric - coerce errors turns invalid values to NaN
            df_ready[col] = pd.to_numeric(val, errors='coerce').fillna(0)

        # --- 4. Scale & Predict ---
        try:
            data_scaled = self.scaler.transform(df_ready)
            dmatrix = xgb.DMatrix(data_scaled, feature_names=self.features)
            prediction = float(self.model.predict(dmatrix)[0])
            
            # Ensure prediction is in valid range [0, 1]
            prediction = max(0.0, min(1.0, prediction))
            return prediction
        except ValueError as e:
            logger.error(f"Value error during prediction: {e}")
            raise ValueError(f"Invalid input data for prediction: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
