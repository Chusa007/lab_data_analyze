# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from scipy.stats import shapiro
from math import sqrt

# Название полей по которым будет строиться датасет
# MSSubClass - Класс объекта
# LotArea - Площадь объекта
# OverallCond - Общее состояние
# TotalBsmtSF - площадь подвала
# GarageArea - площадь гаража
# PoolArea - площадь бассейна
X_NAME = "MSSubClass"
Y_NAME = "LotArea"

# Размер датасета
SIZE_DT_START = 0
SIZE_DT_END = 200

# Масштабирование
SCALE = 10

# Показывать ли графики
SHOW_GRAPH = True


def draw_graph(data_set: pd.core.frame.DataFrame):
    """
    Методя для отрисовки графика
    :param data_set: данные для отрисовки
    """
    data_set.plot(x=X_NAME, y=Y_NAME, style="o")
    plt.xlabel(X_NAME)
    plt.ylabel(Y_NAME)
    plt.show()


def draw_reg(xt: np.ndarray, yt: np.ndarray, yp: np.ndarray, color: str):
    """
    Метод для отрисовки регрессии
    :param xt: сырые данные по x
    :param yt: сырые данные по y
    :param yp: спрогнозированные данне по y
    :param color: цвет линний
    """
    plt.scatter(xt, yt)
    plt.plot(xt, yp, color=color, linewidth="2")
    plt.xlabel(X_NAME)
    plt.ylabel(Y_NAME)
    plt.show()


def scale_array(arr: np.ndarray):
    """
    Метод для масштабирования массива
    :param arr: данные, которые необходимо масштабировать
    :return: arr[i] / SCALE. Уменьшит каждое значение массива на коэф. масштабирования
    """
    return np.where(arr, arr / SCALE, arr)


# Чтение данных из файла
ds = pd.read_csv("train.csv")[SIZE_DT_START:SIZE_DT_END]

# Приводим к виду [[], [], []]. Пример: было [1, 2, 3] стало [[1], [2], [3]]
x = ds[X_NAME].values.reshape(-1, 1)
y = ds[Y_NAME].values.reshape(-1, 1)

# Нормализация данных
y = normalize(y, axis=0).ravel()

# Рисуем по сырым данным
if SHOW_GRAPH:
    draw_graph(ds)

# Проверка на нормальность
statistic, pvalue = shapiro(y)
print(round(statistic, 2), pvalue, 2)

# Разделение данных
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Обучение модели
# LinearRegression()
regressor = Lasso(alpha=0.1)
regressor.fit(x_train, y_train)

# Коэф. регрессии
# print("regressor.intercept_ = {0}, regressor.coef_ = {1}".format(regressor.intercept_, regressor.coef_))

# Построение прогноза
y_pred = regressor.predict(x_test)
df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})

# Посмотреть на справнение прогноза и реальных данных
if SHOW_GRAPH:
    df1 = df.head(10)
    df1.plot(kind="bar")
    plt.show()

# Отобразить нормализованные данные
if SHOW_GRAPH:
    draw_reg(x_test, y_test, y_pred, "green")

# Масштабирование
# draw_reg(scale_array(x_test), scale_array(y_test), scale_array(y_pred), "red")

# Среднеквадратичная ошибка
err = sqrt(mean_squared_error(x_test, y_pred))
print("Среднеквадратичная ошибка = {}".format(err))
