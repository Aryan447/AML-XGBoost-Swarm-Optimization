from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def evaluate(model, dtrain, dtest, y_train, y_test):
    pred_test = model.predict(dtest)
    pred_test_bin = (pred_test > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, pred_test_bin),
        "f1": f1_score(y_test, pred_test_bin),
        "auc": roc_auc_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test_bin),
        "recall": recall_score(y_test, pred_test_bin)
    }
