from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import accuracy_score
import numpy as np


def train_ml_model(model, x_train: np.ndarray, y_train: np.ndarray):
    """Returns a simple machine learning fitted model"""
    return model.fit(x_train, y_train)


def test_ml_model(model, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Returns accuracy on test data"""
    return accuracy_score(y_test, model.predict(x_test))


def build_RFC() -> RFC:
    return RFC(n_estimators=100)


def build_KNC() -> KNC:
    return KNC()
