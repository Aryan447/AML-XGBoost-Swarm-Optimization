"""
Training script for AML detection model using XGBoost with GWO-optimized hyperparameters.

This script:
1. Loads and preprocesses transaction data
2. Balances the dataset
3. Trains an XGBoost model with optimized hyperparameters
4. Saves model artifacts for deployment

Usage:
    python scripts/train.py --data_path /path/to/data.csv --output_dir ./models
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# GWO-optimized hyperparameters
GWO_PARAMS = {
    'eta': 0.012015,              # Learning rate
    'max_depth': 5,                # Depth of trees
    'subsample': 0.945260,         # Data sampling ratio
    'colsample_bytree': 0.860255,  # Feature sampling ratio
    'alpha': 2.558042,             # L1 Regularization
    'lambda': 4.882846,            # L2 Regularization
    'tree_method': 'hist',         # Optimized for speed
    'device': 'cuda'               # Enable GPU if available
}


def load_and_preprocess(data_path: str) -> Tuple[pd.DataFrame, pd.Series, MinMaxScaler]:
    """
    Load and preprocess transaction data.
    
    Args:
        data_path: Path to CSV file containing transaction data
        
    Returns:
        Tuple of (features DataFrame, target Series, fitted scaler)
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}") from e

    # Rename target column if needed
    if 'Is Laundering' in df.columns:
        df.rename(columns={'Is Laundering': 'Is_Laundering'}, inplace=True)
    
    if 'Is_Laundering' not in df.columns:
        raise ValueError("Target column 'Is_Laundering' or 'Is Laundering' not found")

    # Encode categorical features
    logger.info("Encoding categorical features...")
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Payment Format':
            lbl = LabelEncoder()
            df[col] = lbl.fit_transform(df[col].astype(str))

    # Map Payment Format consistently
    payment_map = {
        'Cash': 1, 'Cheque': 2, 'ACH': 3, 'Credit Card': 4,
        'Wire': 5, 'Bitcoin': 6, 'Reinvestment': 7
    }
    if 'Payment Format' in df.columns:
        df['Payment Format'] = df['Payment Format'].map(payment_map).fillna(0).astype(int)

    # Convert timestamp to numeric
    if 'Timestamp' in df.columns and df['Timestamp'].dtype == 'O':
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp']).astype(int) / 10**9
        except Exception as e:
            logger.warning(f"Failed to parse timestamps: {e}")

    # Split features and target
    y = df['Is_Laundering']
    X = df.drop(columns=['Is_Laundering'])

    # Scale features
    logger.info("Scaling features...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    X_final = pd.DataFrame(X_scaled, columns=X.columns)

    logger.info(f"Preprocessed data: {len(X_final)} samples, {len(X_final.columns)} features")
    return X_final, y, scaler


def balance_data(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance dataset by undersampling majority class.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (balanced features, balanced target)
    """
    logger.info("Balancing dataset (undersampling majority class)...")
    data = X.copy()
    data['Is_Laundering'] = y.values

    fraud = data[data['Is_Laundering'] == 1]
    non_fraud = data[data['Is_Laundering'] == 0]

    if len(fraud) == 0:
        raise ValueError("No positive samples (fraud) found in dataset")
    
    if len(non_fraud) == 0:
        raise ValueError("No negative samples (non-fraud) found in dataset")

    # Balanced 50/50 split
    n_samples = min(len(fraud), len(non_fraud))
    non_fraud_sample = non_fraud.sample(n=n_samples, random_state=random_state)
    balanced = pd.concat([fraud, non_fraud_sample]).sample(frac=1, random_state=random_state)

    logger.info(f"Balanced dataset: {len(balanced)} samples (50% fraud, 50% non-fraud)")
    return balanced.drop(columns=['Is_Laundering']), balanced['Is_Laundering']


def train_model(X_train: pd.DataFrame, y_train: pd.Series, num_boost_round: int = 100) -> xgb.Booster:
    """
    Train XGBoost model with optimized hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        num_boost_round: Number of boosting rounds
        
    Returns:
        Trained XGBoost model
    """
    logger.info(f"Training XGBoost model with {num_boost_round} boosting rounds...")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        **GWO_PARAMS,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    try:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False
        )
        logger.info("Model training completed successfully")
        return model
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}") from e


def save_artifacts(
    model: xgb.Booster,
    scaler: MinMaxScaler,
    features: list,
    output_dir: str,
    create_zip: bool = False
) -> None:
    """
    Save model artifacts to disk.
    
    Args:
        model: Trained XGBoost model
        scaler: Fitted scaler
        features: List of feature column names
        output_dir: Directory to save artifacts
        create_zip: Whether to create a zip archive
    """
    logger.info(f"Saving artifacts to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "best_model_gwo.json")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    features_path = os.path.join(output_dir, "feature_columns.pkl")
    
    try:
        model.save_model(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(features, features_path)
        logger.info("Artifacts saved successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to save artifacts: {e}") from e
    
    if create_zip:
        try:
            zip_path = shutil.make_archive('aml_artifacts', 'zip', output_dir)
            logger.info(f"Created archive: {zip_path}")
        except Exception as e:
            logger.warning(f"Failed to create zip archive: {e}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train AML detection model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/kaggle/input/ibm-aml-small/ibm_aml_small.csv",
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save model artifacts"
    )
    parser.add_argument(
        "--num_boost_round",
        type=int,
        default=100,
        help="Number of boosting rounds"
    )
    parser.add_argument(
        "--create_zip",
        action="store_true",
        help="Create zip archive of artifacts"
    )
    
    args = parser.parse_args()
    
    try:
        # Load and preprocess data
        X, y, scaler = load_and_preprocess(args.data_path)
        
        # Balance dataset
        X_train, y_train = balance_data(X, y)
        
        # Train model
        model = train_model(X_train, y_train, args.num_boost_round)
        
        # Save artifacts
        save_artifacts(
            model,
            scaler,
            X_train.columns.tolist(),
            args.output_dir,
            args.create_zip
        )
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
