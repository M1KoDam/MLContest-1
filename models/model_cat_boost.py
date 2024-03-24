from catboost import CatBoostClassifier
import numpy as np
from sklearn.metrics import roc_auc_score


def fit_cat_boost(model, x_train: np.ndarray, y_train: np.ndarray):
    """Returns a cat boost fitted model"""
    return model.fit(x_train, y_train, verbose=False)


def predict_cat_boost(model, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Returns accuracy on test data"""
    roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    return roc_auc


def build_cat_boost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.001,
        depth=10,
    )
