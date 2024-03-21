import numpy as np
import pandas as pd


class DataHandler:
    def __init__(self):
        self.__useless_features = set()
        self.__train_features_count = None

    def handle_train_features(self, train_features: pd.DataFrame) -> np.ndarray:
        train_features = self.__delete_useless_features_and_nans(self.__handle_pandas_dataframe(train_features))
        return self.__handle_numpy_matrix(train_features.values)

    def handle_test_features(self, test_features: pd.DataFrame) -> np.ndarray:
        test_features = self.__delete_useless_features_and_nans(test_features)
        return self.__handle_numpy_matrix(test_features.values)

    def __delete_useless_features_and_nans(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        print(self.__useless_features)
        for uf in self.__useless_features:
            dataframe = dataframe.drop(uf, axis=1)

        # заменяем все nan на 0
        dataframe.fillna(value=0, inplace=True)
        return dataframe

    def __handle_pandas_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        row_count = dataframe.shape[0]

        for column in dataframe.columns.tolist():
            # отсеиваем nan
            if dataframe[column].count() / row_count <= 0.05:
                self.__useless_features.add(column)
                continue

            # отсеиваем строковые хэши
            max_value_count = dataframe[column].value_counts().values.max()
            if max_value_count / row_count <= 0.05: # dataframe[column].dtype == object
                self.__useless_features.add(column)

        return dataframe

    def __handle_numpy_matrix(self, numpy_matrix: np.ndarray) -> np.ndarray:
        column = numpy_matrix.shape[0]
        row = numpy_matrix.shape[1]

        new_matrix = np.zeros(numpy_matrix.shape, dtype='float64')

        for r in range(row):
            values_counter = {}
            next_el_number = 1

            for c in range(column):
                cur_value = numpy_matrix[c][r]
                target_value = cur_value
                # если значение число - оставляем его. NaN меняем на 0
                if isinstance(cur_value, (int, float)):
                    if np.isnan(cur_value):
                        target_value = 0
                # если значение не число - то предполагаем что это строка представляющая собой характеристику
                elif cur_value in values_counter.keys():
                    target_value = values_counter[cur_value]
                else:
                    target_value = next_el_number
                    values_counter[cur_value] = next_el_number
                    next_el_number += 1
                new_matrix[c][r] = target_value

        return new_matrix


def handle_labels(labels):
    labels[labels == -1] = 0
    return labels.astype('float32')


def get_classes(y: np.ndarray):
    return np.round(y)


def handle_features(train_features: pd.DataFrame, test_features: pd.DataFrame):
    dh = DataHandler()
    x_train = dh.handle_train_features(train_features)
    x_test = dh.handle_test_features(test_features)

    return x_train, x_test
