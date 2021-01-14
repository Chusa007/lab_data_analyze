# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

# Переменная для хранения флага отображения графиков
SHOW_GRAPH = False


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


def build_model(data_set: pd.core.frame.DataFrame, data: np.ndarray) -> dict:
    """
    Метод для обучения модели
    :param data_set: сырые данные
    :param data: данные
    :return: метод возвращает построенную модель
    """
    # Обучение модели классификации
    # Разделение данных
    x_train, x_test, y_train, y_test = train_test_split(data_set, data, stratify=data, test_size=0.33, random_state=0)
    # Создание модели
    model = GradientBoostingClassifier(n_estimators=100, random_state=0)
    # model = AdaBoostClassifier(n_estimators=100, random_state=0)
    # model = RandomForestClassifier(n_estimators=100, random_state=0, max_features='sqrt', n_jobs=-1, verbose=1)

    # Обучение модели
    model.fit(x_train, y_train)

    # Построение прогноза
    y_pred = model.predict(x_test)

    if SHOW_GRAPH:
        # Построение прогноза
        df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.flatten()})

        # Посмотреть на справнение прогноза и реальных данных
        df1 = df.head(20)
        df1.plot(kind="bar")
        plt.show()

    f1_res = f1_score(y_test, y_pred, average='micro')

    return dict({"y_test": y_test, "y_pred": y_pred, "f1_res": f1_res})

# Чтение данных из файла
ds = pd.read_csv("Development Index.csv")
y = np.array(ds.pop("Development Index"))


# first = y[:45]
# second = y[45:90]
# medium = y[90:135]
# pre_last = y[135:180]
# last = y[180:]


# ds_without_last = ds[:180]
# y_without_last = [*first, *second, *medium, *pre_last]
#
# ds_without_pre_last = ds[lambda x: np.logical_or(x.index < 135, x.index >= 180)]
# y_without_pre_last = [*first, *second, *medium, *last]
#
# ds_without_medium = ds[lambda x: np.logical_or(x.index < 90, x.index >= 135)]
# y_without_medium = [*first, *second, *pre_last, *last]
#
# ds_without_second = ds[lambda x: np.logical_or(x.index < 45, x.index >= 90)]
# y_without_second = [*first, *medium, *pre_last, *last]
#
# ds_without_first = ds[45:]
# y_without_first = [*second, *medium, *pre_last, *last]
#
# ds_pair = [{"ds": ds_without_last, "y": y_without_last},
#            {"ds": ds_without_pre_last, "y": y_without_pre_last},
#            {"ds": ds_without_medium, "y": y_without_medium},
#            {"ds": ds_without_second, "y": y_without_second},
#            {"ds": ds_without_first, "y": y_without_first}]


first = y[:75]
medium = y[75:150]
last = y[150:]

ds1_2 = ds[:150]
y1_2 = [*first, *medium]

ds1_3 = ds[lambda x: np.logical_or(x.index < 75, x.index >= 150)]
y1_3 = [*last, *first]

ds2_3 = ds[75:]
y2_3 = [*medium, *last]

ds_pair = [{"ds": ds1_2, "y": y1_2},
           {"ds": ds1_3, "y": y1_3},
           {"ds": ds2_3, "y": y2_3}]




f1_sum = 0
sum_err = {"false_positive": [0, 0, 0, 0], "false_negative": [0, 0, 0, 0]}

i = 1
for dct in ds_pair:
    # Строим модель
    data = build_model(dct.get("ds"), dct.get("y"))
    f1_sum += data.get("f1_res")
    res = count_errors(data)
    print("Model №{num}\nf1 = {f1}\n{res}".format(num=i, res=res, f1=data.get("f1_res")))

    for key, value in sum_err.items():
        sum_err[key] = [i + j for i, j in zip(res[key], value)]
    i += 1

for key, value in sum_err.items():
    sum_err[key] = [round(i / len(ds_pair), 2) for i in value]

print("\nСумма ошибок 1 и 2 рода по всем моделям:\n{sum}\nf1 average = {f1}".format(sum=sum_err, f1=round(f1_sum / len(ds_pair), 2)))

