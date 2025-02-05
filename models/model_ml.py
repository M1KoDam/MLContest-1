from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


def fit_ml_model(model, x_train: np.ndarray, y_train: np.ndarray):
    """Returns a simple machine learning fitted model"""
    return model.fit(x_train, y_train)


def predict_ml_model(model, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Returns accuracy on test data"""
    roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    return roc_auc


def build_RFC() -> RFC:
    return RFC(n_estimators=1000)


def build_KNC() -> KNC:
    return KNC()
