# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Переменная для хранения флага отображения графиков
SHOW_GRAPH = True


def comparison_list(test: list, pred: list) -> tuple:
    """
    Метод для сравнения масштабированных массивов
    :param test: тестовые данные
    :param pred: предсказанные данные
    :return: ошибки 1 и 2 рода
    """
    f_p = 0
    f_n =0

    for i in range(len(test)):
        if test[i] == pred[i]:
            continue
        elif test[i] > pred[i]:
            f_n += 1
        else:
            f_p += 1

    return f_p, f_n


def count_errors(data: dict) -> dict:
    """
    Подсчет ошибок 1 и 2 рода
    :param data: словарь с тестовыми данными и предсказанные данные
    :return: возвращает количество ошибок 1 и 2 рода
    """
    y_test = data.get("y_test")
    y_pred = data.get("y_pred")
    class_array = np.unique(y_test)
    # Массивы ошибок 1 и 2 рода по классам
    false_positive = []
    false_negative = []

    for cl in class_array:
        # Масштабированные массивы
        y_test_scaling = []
        y_pred_scaling = []

        # Масштабируем
        for elem in y_test:
            y_test_scaling.append(1 if cl == elem else 0)
        for elem in y_pred:
            y_pred_scaling.append(1 if cl == elem else 0)

        # Считаем ошибки
        fp, fn = comparison_list(y_test_scaling, y_pred_scaling)
        false_positive.append(fp)
        false_negative.append(fn)

    return dict({"false_positive": false_positive, "false_negative": false_negative})


def build_model(data_set: pd.core.frame.DataFrame, data: np.ndarray, test_size: float) -> dict:
    """
    Метод для обучения модели
    :param data_set: сырые данные
    :param data: данные
    :param test_size: размер данных
    :return: метод возвращает построенную модель
    """
    # Обучение модели классификации
    # Разделение данных
    x_train, x_test, y_train, y_test = train_test_split(data_set, data, stratify=data, test_size=test_size, random_state=0)

    # Создание модели
    model = AdaBoostClassifier(n_estimators=100, random_state=0)
    # model = RandomForestClassifier(n_estimators=100, random_state=0, max_features='sqrt', n_jobs=-1, verbose=1)

    # Обучение модели
    model.fit(x_train, y_train)

    # Построение прогноза
    y_pred = model.predict(x_test)

    if SHOW_GRAPH:
        # Построение прогноза
        df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})

        # Посмотреть на справнение прогноза и реальных данных
        df1 = df.head(20)
        df1.plot(kind="bar")
        plt.show()

    return dict({"y_test": y_test, "y_pred": y_pred})

# Чтение данных из файла
ds = pd.read_csv("Development Index.csv")
y = np.array(ds.pop("Development Index"))

# Задаем размер тестовых данных к обучаемым
test_size_arr = [0.33, 0.495, 0.66]

sum_err = {"false_positive": [0, 0, 0, 0], "false_negative": [0, 0, 0, 0]}

for t_size in test_size_arr:
    # Строим модель
    data = build_model(ds, y, t_size)
    res = count_errors(data)
    print("Model №{num}\n{res}".format(num=test_size_arr.index(t_size) + 1, res=res))

    for key, value in sum_err.items():
        sum_err[key] = [i + j for i, j in zip(res[key], value)]


print("\nСумма ошибок 1 и 2 рода по всем моделям:\n{sum}".format(sum=sum_err))

