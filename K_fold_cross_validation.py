import numpy as np
from data_handler import get_data
from models.model_keras import build_keras_model, train_keras_model, test_keras_model
from models.model_ml import build_RFC, train_ml_model, test_ml_model, build_KNC, build_cat_boost
import pandas as pd


def K_fold_cross_validation(train_data: np.ndarray,
                            train_targets: np.ndarray,
                            build_model_func,
                            train_model_func,
                            test_model_func,
                            k=4,
                            verbose=True) -> float:
    num_val_samples = len(train_data) // k
    roc_auc_scores = []

    for i in range(k):
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                             train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

        model = build_model_func()
        trained_model = train_model_func(model, partial_train_data, partial_train_targets)
        test_roc_auc = test_model_func(trained_model, val_data, val_targets)
        if verbose:
            print(f"Iteration {i+1}/{k}: ROC AUC: {test_roc_auc}")
        roc_auc_scores.append(test_roc_auc)

    return sum(roc_auc_scores) / len(roc_auc_scores)


def main():
    x_train, y_train, x_test = get_data()
    print(x_train.shape)
    print(len(y_train[y_train == 1]))

    name = "CAT"
    roc_auc_score = K_fold_cross_validation(x_train, y_train, build_cat_boost, train_ml_model, test_ml_model)
    print("Result ROC AUC:", roc_auc_score)
    print("Start training:")
    print(x_test.shape)
    model_ml = build_cat_boost()
    model_ml.fit(x_train, y_train)
    y_pred = model_ml.predict_proba(x_test)[:, 1]

    # name = "Keras"
    # print("Result ROC AUC:", K_fold_cross_validation(x_train, y_train, build_keras_model, train_keras_model, test_keras_model))
    # print("Start training:")
    # model = build_keras_model()
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test).flatten()

    submission = pd.DataFrame({'ID': pd.read_csv(f"data/data_predict.csv")['ID'].to_list(), 'Target': y_pred})
    print(submission)
    submission.to_csv(f"data/data_predict_by_{name}.csv", index=False)


if __name__ == "__main__":
    main()
