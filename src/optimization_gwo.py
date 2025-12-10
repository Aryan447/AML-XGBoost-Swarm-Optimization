from mealpy import FloatVar, GWO
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score

def optimize_gwo(dtrain, dtest, y_test):
    def gwo_obj(sol):
        params = {
            'objective':'binary:logistic','eval_metric':'logloss',
            'eta':sol[0],'max_depth':int(sol[1]),
            'subsample':sol[2],'colsample_bytree':sol[3],
            'alpha':sol[4],'lambda':sol[5],
            'device':'cuda','tree_method':'hist'
        }
        bst = xgb.train(params, dtrain, 100, evals=[(dtest,"test")], verbose_eval=False)
        pred = np.round(bst.predict(dtest))
        return -f1_score(y_test, pred)

    problem = {
        "obj_func": gwo_obj,
        "bounds": FloatVar(lb=[0.01,3,0.5,0.5,0.0,0.0], ub=[0.3,20,1,1,10,10]),
        "minmax":"min",
    }

    gwo = GWO.OriginalGWO(epoch=100, pop_size=75)
    best = gwo.solve(problem)
    return best.solution
