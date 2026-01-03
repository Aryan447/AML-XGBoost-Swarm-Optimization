"""Model service for loading and using the ONNX model."""
import os
import logging
import json
import tempfile
from typing import Optional, Dict, List, Any, Union
from urllib.request import urlretrieve
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing and using the AML detection model via ONNX."""
    
    def __init__(self, model_dir: str) -> None:
        """
        Initialize the model service and load artifacts.
        
        Args:
            model_dir: Directory path containing model artifacts
        """
        self.session: Optional[ort.InferenceSession] = None
        self.scaler_params: Optional[Dict] = None
        self.features: Optional[List[str]] = None
        self._load_artifacts(model_dir)

    def _load_artifacts(self, model_dir: str) -> None:
        """
        Load model artifacts from the specified directory.
        """
        try:
            # Handle URL vs Local Path
            is_url = model_dir.startswith(('http://', 'https://'))
            
            if is_url:
                logger.info(f"Loading models from URL: {model_dir}...")
                temp_dir = tempfile.mkdtemp()
                model_path = os.path.join(temp_dir, "model.onnx")
                scaler_path = os.path.join(temp_dir, "scaler_params.json")
                
                urlretrieve(f"{model_dir}/model.onnx", model_path)
                urlretrieve(f"{model_dir}/scaler_params.json", scaler_path)
            else:
                logger.info(f"Loading models from local path: {model_dir}...")
                model_path = os.path.join(model_dir, "model.onnx")
                scaler_path = os.path.join(model_dir, "scaler_params.json")

            # Verify files
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler params not found at {scaler_path}")

            # Load ONNX Session
            self.session = ort.InferenceSession(model_path)
            
            # Load Scaler Params
            with open(scaler_path, 'r') as f:
                self.scaler_params = json.load(f)
                
            self.features = self.scaler_params.get("feature_names", [])
            
            logger.info(f"Model loaded. Features: {len(self.features)}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise RuntimeError(f"Failed to load model artifacts: {e}")

    def predict(self, input_data: Dict[str, Any]) -> float:
        """
        Predict risk scorer for a transaction dictionary.
        """
        if not self.session or not self.scaler_params:
            raise RuntimeError("Model service not initialized.")

        try:
            # 1. Preprocess
            features_array = self._preprocess(input_data)
            
            # 2. Predict (ONNX)
            input_name = self.session.get_inputs()[0].name
            # ONNX Runtime expects float32
            feed_dict = {input_name: features_array.astype(np.float32)}
            
            predictions = self.session.run(None, feed_dict)
            
            # Output is usually [probabilities] or [label, probabilities] depending on conversion
            # XGBoost classifier via onnxmltools usually outputs [label, probabilities_tensor]
            
            # Let's inspect the output structure safely
            # Usually index 1 contains a list of dictionaries (zipmap) or a tensor of probabilities
            raw_pred = predictions[1] 
            
            score = 0.0
            if isinstance(raw_pred, list) and isinstance(raw_pred[0], dict):
                 # ZipMap output: [{'0': 0.9, '1': 0.1}, ...]
                 score = raw_pred[0].get(1, 0.0)
            elif isinstance(raw_pred, np.ndarray):
                # Tensor output: [[0.9, 0.1]]
                # Assuming index 1 is the positive class
                if raw_pred.shape[1] > 1:
                    score = float(raw_pred[0][1])
                else:
                    score = float(raw_pred[0][0])
            
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Prediction failed: {e}")

    def _preprocess(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Manually preprocess input dictionary to scaled numpy array.
        """
        # 1. Parse Timestamp
        timestamp = 0.0
        if 'Timestamp' in data:
            # simplified parsing, assuming input might already be cleaned or simple format
            # In a real scenario without pandas, we'd use datetime.strptime
            try:
                # If it's already a float/int
                timestamp = float(data['Timestamp'])
            except:
                 # Fallback for now or implement strptime if format is known
                 timestamp = 0.0

        # 2. Encode Categorical (Payment Format)
        pay_map = {
            'Cash': 1, 'Cheque': 2, 'ACH': 3, 'Credit Card': 4,
            'Wire': 5, 'Bitcoin': 6, 'Reinvestment': 7
        }
        pay_fmt = pay_map.get(data.get('Payment Format'), 0)

        # 3. Construct Feature Vector
        vector = []
        for feat in self.features:
            val = 0.0
            if feat == 'Timestamp':
                val = timestamp
            elif feat == 'Payment Format':
                val = float(pay_fmt)
            elif feat in data:
                try:
                    val = float(data[feat])
                except:
                    val = 0.0
            
            vector.append(val)
        
        # 4. Scale
        X = np.array([vector], dtype=np.float32)
        
        if self.scaler_params.get("type") == "MinMaxScaler":
            min_val = np.array(self.scaler_params['min'], dtype=np.float32)
            scale_val = np.array(self.scaler_params['scale'], dtype=np.float32)
            X_scaled = X * scale_val + min_val
        else:
            # Assume StandardScaler
            mean = np.array(self.scaler_params['mean'], dtype=np.float32)
            scale = np.array(self.scaler_params['scale'], dtype=np.float32)
            X_scaled = (X - mean) / scale
            
        return X_scaled

