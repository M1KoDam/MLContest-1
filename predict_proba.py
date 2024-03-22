from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from data_handler import get_data, handle_labels
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np


if __name__ == "__main__":
    x_train, y_train, x_test = get_data()

    print(x_train.shape)
    print(x_test.shape)

    print("y_train:", len(y_train), np.count_nonzero(y_train))

    model = RFC(n_estimators=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_test)[:, 1]
    print(y_pred)

    model65 = RFC(min_samples_split=2, min_samples_leaf=1, n_estimators=100)
    model65.fit(x_train, y_train)
    y_pred65 = model65.predict_proba(x_test)[:, 1]
    print(y_pred65)

    submission = pd.DataFrame({'ID': pd.read_csv("data/data_predict.csv")['ID'].to_list(), 'Target': y_pred})
    print(submission)
    submission.to_csv('data/result_submission.csv', index=False)

    # print("y_pred:", len(y_pred), np.count_nonzero(y_pred))
    # print("Roc auc:", roc_auc_score(y_pred.astype(int), y_roc_auc)) # а пошёл ты нахуй