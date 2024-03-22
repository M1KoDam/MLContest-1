import numpy as np
import pandas as pd


class DataHandler:
    def __init__(self, handle_pandas=True, normalize=False):
        self.__handle_pandas = handle_pandas
        self.__normalize = normalize
        self.__useless_features = set()
        self.__train_features_count = None

    def handle_train_features(self, train_features: pd.DataFrame) -> np.ndarray:
        train_features = train_features.sample(frac=1)
        train_features = self.__delete_useless_features_and_nans(self.__handle_pandas_dataframe(train_features))
        return self.__normalize_data(self.__handle_numpy_matrix(train_features.values))

    def handle_test_features(self, test_features: pd.DataFrame) -> np.ndarray:
        test_features = self.__delete_useless_features_and_nans(test_features)
        return self.__normalize_data(self.__handle_numpy_matrix(test_features.values))

    def __normalize_data(self, data: np.ndarray) -> np.ndarray:
        if self.__normalize:
            data -= data.mean(axis=0)
            data /= data.std(axis=0)
            data = np.nan_to_num(data, nan=0)
        return data

    def __delete_useless_features_and_nans(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # print(self.__useless_features)
        for uf in self.__useless_features:
            dataframe = dataframe.drop(uf, axis=1)

        # заменяем все nan на 0
        dataframe.fillna(value=0, inplace=True)
        return dataframe

    def __handle_pandas_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        row_count = dataframe.shape[0]

        if self.__handle_pandas:
            for column in dataframe.columns.tolist():
                # отсеиваем nan
                if dataframe[column].count() / row_count <= 0.05:
                    self.__useless_features.add(column)
                    continue

                # отсеиваем строковые хэши
                max_value_count = dataframe[column].value_counts().values.max()
                if dataframe[column].dtype == object and max_value_count / row_count <= 0.05:  #  dataframe[column].dtype == object
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


def get_data():
    dh = DataHandler(
        handle_pandas=True,
        normalize=True
    )

    # Достаём из csv
    train_data = pd.read_csv("data/train.csv")
    train_labels_pd = train_data['Target'].values
    train_features_pd = train_data.drop(columns=['Target', 'ID'])

    test_features_pd = pd.read_csv("data/data_predict.csv").drop(columns=['ID'])

    # Обрабатываем
    x_train = dh.handle_train_features(train_features_pd)
    x_test = dh.handle_test_features(test_features_pd)
    y_train = handle_labels(train_labels_pd)

    return x_train, y_train, x_test
