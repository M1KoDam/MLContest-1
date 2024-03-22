import numpy as np
from data_handler import get_data
from models.model_keras import build_keras_model, train_keras_model, test_keras_model
from models.model_ml import build_RFC, train_ml_model, test_ml_model, build_KNC
import pandas as pd


def K_fold_cross_validation(train_data: np.ndarray,
                            train_targets: np.ndarray,
                            model,
                            train_model_func,
                            test_model_func,
                            k=4) -> float:
    num_val_samples = len(train_data) // k
    accuracy_scores = []

    for i in range(k):
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                             train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

        trained_model = train_model_func(model, partial_train_data, partial_train_targets)
        test_acc = test_model_func(trained_model, val_data, val_targets)
        accuracy_scores.append(test_acc)

    return sum(accuracy_scores) / len(accuracy_scores)


def main():
    x_train, y_train, x_test = get_data()
    print(x_train.shape)
    # print(K_fold_cross_validation(x_train, y_train, build_RFC(), train_ml_model, test_ml_model))
    # print(K_fold_cross_validation(x_train, y_train, build_KNC(), train_ml_model, test_ml_model))

    model_ml = build_RFC()
    name = "RFC"
    print(K_fold_cross_validation(x_train, y_train, build_RFC(), train_ml_model, test_ml_model))
    y_pred = model_ml.predict_proba(x_test)[:, 1]

    # model_keras = build_keras_model()
    # name = "Keras"
    # print(K_fold_cross_validation(x_train, y_train, model_keras, train_keras_model, test_keras_model))
    # y_pred = model_keras.predict(x_test).flatten()

    submission = pd.DataFrame({'ID': pd.read_csv(f"data/data_predict_by_{name}.csv")['ID'].to_list(), 'Target': y_pred})
    print(submission)
    submission.to_csv('data/result_submission.csv', index=False)


if __name__ == "__main__":
    main()
