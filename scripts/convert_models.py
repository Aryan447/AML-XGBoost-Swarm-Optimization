
import os
import joblib
import json
import numpy as np
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType as ONNXFloat
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "model.onnx")
SCALER_METADATA_PATH = os.path.join(MODELS_DIR, "scaler_params.json")

def convert_models():
    logger.info("Starting model conversion...")
    
    # Load existing artifacts
    try:
        model_path = os.path.join(MODELS_DIR, "best_model_gwo.json")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        features_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
        
        logger.info(f"Loading XGBoost model from {model_path}")
        
        # FIX: Patch model JSON to fix base_score format issue
        with open(model_path, 'r') as f:
            model_json = json.load(f)
        
        # Patches for onnxmltools compatibility
        if "learner" in model_json and "learner_model_param" in model_json["learner"]:
             # Force base_score to be a string float not a list string like "[5E-1]"
             bs = model_json["learner"]["learner_model_param"].get("base_score")
             if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                 logger.info(f"Patching base_score: {bs} -> {bs[1:-1]}")
                 model_json["learner"]["learner_model_param"]["base_score"] = bs[1:-1]

        # Save patched model temporarily
        patched_model_path = os.path.join(MODELS_DIR, "temp_patched_model.json")
        with open(patched_model_path, 'w') as f:
            json.dump(model_json, f)

        bst = xgb.Booster()
        bst.load_model(patched_model_path)
        
        logger.info(f"Loading Scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        logger.info(f"Loading Features from {features_path}")
        features = joblib.load(features_path)
        num_features = len(features)
        
    except FileNotFoundError as e:
        logger.error(f"Error loading files: {e}")
        return
    
    # Monkeypatch xgb.Booster.save_config to fix base_score issue for onnxmltools
    def patched_save_config(self):
        config_str = original_save_config(self)
        try:
            config = json.loads(config_str)
            if "learner" in config and "learner_model_param" in config["learner"]:
                bs = config["learner"]["learner_model_param"].get("base_score")
                # Fix [5E-1] -> 0.5
                if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                    config["learner"]["learner_model_param"]["base_score"] = bs[1:-1]
            return json.dumps(config)
        except Exception as e:
            logger.warning(f"Monkeypatch failed: {e}")
            return config_str

    original_save_config = xgb.Booster.save_config
    xgb.Booster.save_config = patched_save_config

    # --- Convert XGBoost to ONNX ---
    logger.info("Converting XGBoost model to ONNX...")
    
    # Strip feature names to avoid "Unable to interpret..." error
    # onnxmltools prefers f0, f1 style or indices when mapping directly
    bst.feature_names = None
    bst.feature_types = None
    
    initial_type = [('float_input', ONNXFloat([None, num_features]))]
    onnx_model = onnxmltools.convert.convert_xgboost(bst, initial_types=initial_type)
    
    with open(ONNX_MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info(f"Saved ONNX model to {ONNX_MODEL_PATH}")
    
    # --- Extract Scaler Parameters ---
    logger.info("Extracting scaler parameters...")
    
    # Check scaler type
    if hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
         logger.info("Detected MinMaxScaler")
         scaler_params = {
            "min": scaler.min_.tolist(),
            "scale": scaler.scale_.tolist(),
            "feature_names": features,
            "type": "MinMaxScaler"
        }
    else:
         logger.info("Assuming StandardScaler")
         scaler_params = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "feature_names": features,
            "type": "StandardScaler"
         }
    
    with open(SCALER_METADATA_PATH, "w") as f:
        json.dump(scaler_params, f)
    logger.info(f"Saved scaler parameters to {SCALER_METADATA_PATH}")

    logger.info("Conversion complete!")

if __name__ == "__main__":
    convert_models()
