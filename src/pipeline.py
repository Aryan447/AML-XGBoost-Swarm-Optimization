from data_loader import load_data
from preprocess import preprocess_data, balance_data
from model_train import prepare_train_test
from optimization_gwo import optimize_gwo
from save_artifacts import save_artifacts
import xgboost as xgb

df = load_data("HI-Small_Trans.csv")
df_scaled, scaler = preprocess_data(df)
df_balanced = balance_data(df_scaled)

X_train, X_test, y_train, y_test, dtrain, dtest = prepare_train_test(df_balanced)

best_params = optimize_gwo(dtrain, dtest, y_test)

params = {
    'objective':'binary:logistic','eval_metric':'logloss',
    'eta':best_params[0],
    'max_depth':int(best_params[1]),
    'subsample':best_params[2],
    'colsample_bytree':best_params[3],
    'alpha':best_params[4],
    'lambda':best_params[5],
    'device':'cuda',
    'tree_method':'hist'
}

model = xgb.train(params, dtrain, 100)

save_artifacts(model, scaler, X_train.columns.tolist())
