import numpy as np
import pandas as pd


class DataHandler:
    def __init__(self, handle_pandas=True, normalize=False, stabilize=False, use_most_profitable_features=False):
        self.__handle_pandas = handle_pandas
        self.__normalize = normalize
        self.__stabilize = stabilize
        self.__useless_features = set()
        self.__train_features_count = None
        self.__use_most_profitable_features = use_most_profitable_features
        self.__most_profitable_features = {'Var6': 0.5213117858833046, 'Var7': 0.556139829731122,
                                           'Var13': 0.520777890026955, 'Var28': 0.5416848198157682,
                                           'Var65': 0.5553836702957129, 'Var72': 0.5256595924736195,
                                           'Var73': 0.6048664202551313, 'Var74': 0.5575938752838056,
                                           'Var123': 0.5274721950588497, 'Var125': 0.5308152242948924,
                                           'Var126': 0.660046965946915, 'Var140': 0.5338185970507616,
                                           'Var144': 0.5358285934456871, 'Var189': 0.5718020009743959,
                                           'Var192': 0.5396235490732563, 'Var193': 0.5491077351379352,
                                           'Var197': 0.5226392817047648, 'Var198': 0.543856991792264,
                                           'Var199': 0.5468472370159597, 'Var200': 0.5334267525156573,
                                           'Var202': 0.5277395449579272, 'Var204': 0.5387559182786306,
                                           'Var205': 0.5434843719054118, 'Var206': 0.5872481785293483,
                                           'Var207': 0.5464373238435024, 'Var211': 0.5221321858530228,
                                           'Var212': 0.5729484848047722, 'Var214': 0.538342375439889,
                                           'Var216': 0.5739819099299388, 'Var217': 0.5471774859303655,
                                           'Var218': 0.5629230430971279, 'Var220': 0.532774422480478,
                                           'Var221': 0.5371070186199243, 'Var222': 0.5381776870983123,
                                           'Var225': 0.5581703620918544, 'Var226': 0.5397595024999501,
                                           'Var227': 0.5459644348449697, 'Var228': 0.5605729371639254,
                                           'Var229': 0.5622587374100138}

    def handle_train_features(self, train_data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        train_data = train_data.drop_duplicates()
        y_l = train_data['Target']
        min_count = min(len(y_l[y_l == -1]), len(y_l[y_l == 1]))

        new_data = train_data
        if self.__stabilize:
            new_data = train_data.groupby('Target').head(min_count).reset_index(drop=True)
        # Перемешаем данные
        new_data = new_data.sample(frac=1)

        train_labels_pd = new_data['Target']
        train_features_pd = new_data.drop(columns=['Target', 'ID'])
        if self.__use_most_profitable_features:
            train_features_pd = train_features_pd[list(self.__most_profitable_features.keys())]
        y_train = handle_labels(train_labels_pd.values)

        x_train = self.__delete_useless_features_and_nans(self.__handle_pandas_dataframe(train_features_pd)).values
        x_train = self.__normalize_data(self.__handle_numpy_matrix(x_train))
        return x_train, y_train

    def handle_test_features(self, test_features: pd.DataFrame) -> np.ndarray:
        if self.__use_most_profitable_features:
            test_features = test_features[list(self.__most_profitable_features.keys())]
        x_test = self.__delete_useless_features_and_nans(test_features).values
        x_test = self.__normalize_data(self.__handle_numpy_matrix(x_test))
        return x_test

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
                if dataframe[column].dtype == object and max_value_count / row_count <= 0.05:
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


def handle_labels(labels: np.ndarray) -> np.ndarray:
    labels[labels == -1] = 0
    return labels.astype('float32')


def get_data():
    dh = DataHandler(
        handle_pandas=False,
        normalize=False,  # Жизненно важно для нейросетей
        stabilize=False,
        use_most_profitable_features=False
    )

    # Достаём из csv
    train_data = pd.read_csv("data/train.csv")

    test_features_pd = pd.read_csv("data/data_predict.csv").drop(columns=['ID'])

    # Обрабатываем
    x_train, y_train = dh.handle_train_features(train_data)
    x_test = dh.handle_test_features(test_features_pd)

    return x_train, y_train, x_test


if __name__ == "__main__":
    get_data()
