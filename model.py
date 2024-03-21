from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from data_handler import handle_features, handle_labels, get_classes
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np

train_data = pd.read_csv("data/train.csv")
train_labels_pd = train_data['Target'].values
train_features_pd = train_data.drop(columns=['Target', 'ID'])

test_features_pd = pd.read_csv("data/data_predict.csv").drop(columns=['ID'])
test_labels_pd = pd.read_csv("data/sample_submission.csv").drop(columns=['ID']).values

x_train, x_test = handle_features(train_features_pd, test_features_pd)
y_train = handle_labels(train_labels_pd)

y_roc_auc = test_labels_pd
y_test = get_classes(test_labels_pd).flatten()

print(x_train.shape)
print(x_test.shape)

print("y_test:", len(y_test), np.count_nonzero(y_test))
print("y_train:", len(y_train), np.count_nonzero(y_train))

model = GNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("y_pred:", len(y_pred), np.count_nonzero(y_pred))
print("Roc auc:", roc_auc_score(y_pred.astype(int), y_roc_auc))


