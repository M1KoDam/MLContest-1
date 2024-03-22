from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from keras.metrics import AUC


def build_keras_model() -> Sequential:
    model = Sequential()
    model.add(Dense(64, activation='relu'))  # input_shape=(train_data.shape[1],)
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=[AUC()])

    return model


def train_keras_model(model: Sequential, x_train: np.ndarray, y_train: np.ndarray, num_epochs=10,
                      batch_size=128) -> Sequential:
    """Returns a Keras Sequential fitted model"""
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

    return model


def test_keras_model(model: Sequential, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Returns accuracy on test data"""
    # return roc_auc_score(y_test, model.predict(x_test))
    val_loss, val_roc_auc = model.evaluate(x_test, y_test, verbose=1)
    print(val_roc_auc)
    return val_roc_auc
