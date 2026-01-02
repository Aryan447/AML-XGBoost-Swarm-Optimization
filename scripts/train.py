import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ==========================================
# 1. GWO HYPERPARAMETERS
# ==========================================
# Best parameters derived from your GWO optimization run
# (Adjust these if your specific run yielded slightly different results)
gwo_params = {
    'eta': 0.012015,             # Learning rate
    'max_depth': 5,          # Depth of trees
    'subsample': 0.945260,        # Data sampling ratio
    'colsample_bytree': 0.860255, # Feature sampling ratio
    'alpha': 2.558042,           # L1 Regularization (from your results)
    'lambda': 4.882846,           # L2 Regularization
    'tree_method': 'hist',   # Optimized for speed
    'device': 'cuda'         # Enable GPU on Kaggle
}

# ==========================================
# 2. DATA PIPELINE
# ==========================================
def load_and_preprocess(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path)

    # Rename target
    if 'Is Laundering' in df.columns:
        df.rename(columns={'Is Laundering': 'Is_Laundering'}, inplace=True)

    # 1. Feature Engineering / Cleaning
    # Encode Categoricals
    print("Encoding categorical features...")
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Payment Format':
            lbl = LabelEncoder()
            df[col] = lbl.fit_transform(df[col].astype(str))

    # Map Payment Format (Consistent mapping)
    payment_map = {'Cash':1, 'Cheque':2, 'ACH':3, 'Credit Card':4,
                   'Wire':5, 'Bitcoin':6, 'Reinvestment':7}
    if 'Payment Format' in df.columns:
        df['Payment Format'] = df['Payment Format'].map(payment_map).fillna(0).astype(int)

    # Timestamp to numeric
    if df['Timestamp'].dtype == 'O':
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).astype(int) / 10**9

    # Split X/y
    y = df['Is_Laundering']
    X = df.drop(columns=['Is_Laundering'])

    # Scale Features
    print("Scaling features...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    X_final = pd.DataFrame(X_scaled, columns=X.columns)

    return X_final, y, scaler

def balance_data(X, y):
    print("Balancing dataset (Undersampling majority)...")
    data = X.copy()
    data['Is_Laundering'] = y.values

    fraud = data[data['Is_Laundering'] == 1]
    non_fraud = data[data['Is_Laundering'] == 0]

    # Balanced 50/50 split
    non_fraud_sample = non_fraud.sample(n=len(fraud), random_state=42)
    balanced = pd.concat([fraud, non_fraud_sample]).sample(frac=1, random_state=42)

    return balanced.drop(columns=['Is_Laundering']), balanced['Is_Laundering']

# ==========================================
# 3. EXECUTION
# ==========================================
DATA_PATH = "/kaggle/input/ibm-aml-small/ibm_aml_small.csv"

# Run Pipeline
X, y, scaler = load_and_preprocess(DATA_PATH)
X_train, y_train = balance_data(X, y)

# Train
print("Training XGBoost Model...")
dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(
    {**gwo_params, 'objective': 'binary:logistic', 'eval_metric': 'logloss'},
    dtrain,
    num_boost_round=100
)

# Save Artifacts
print("Saving artifacts...")
os.makedirs("models", exist_ok=True)
model.save_model("models/best_model_gwo.json")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")

# Zip for download
shutil.make_archive('aml_artifacts', 'zip', 'models')
print("DONE! Download 'aml_artifacts.zip' from Output.")
