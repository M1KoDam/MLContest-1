import numpy as np
from data_handler import get_data
from models.model_keras import build_keras_model, train_keras_model, test_keras_model
from models.model_ml import build_RFC, train_ml_model, test_ml_model, build_KNC


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
    # print(x_train)
    # print(K_fold_cross_validation(x_train, y_train, build_RFC(), train_ml_model, test_ml_model))
    print(K_fold_cross_validation(x_train, y_train, build_KNC(), train_ml_model, test_ml_model))
    # print(K_fold_cross_validation(x_train, y_train, build_keras_model(), train_keras_model, test_keras_model))


if __name__ == "__main__":
    main()
