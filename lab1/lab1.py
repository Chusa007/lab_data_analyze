# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy.stats import shapiro

# Название полей по которым будет строиться датасет
# MSSubClass - Класс объекта
# LotArea - Площадь объекта
# OverallCond - Общее состояние
# TotalBsmtSF - площадь подвала
# GarageArea - площадь гаража
# PoolArea - площадь бассейна
X_NAME = "MSSubClass"
Y_NAME = "TotalBsmtSF"

# Размер датасета
SIZE_DT = 200

# Масштабирование
SCALE = 10


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
    plt.show()


def scale_array(arr: np.ndarray):
    """
    Метод для масштабирования массива
    :param arr: данные, которые необходимо масштабировать
    :return: arr[i] / SCALE. Уменьшит каждое значение массива на коэф. масштабирования
    """
    return np.where(arr, arr / SCALE, arr)


# Чтение данных из файла
ds = pd.read_csv("train.csv")[:SIZE_DT]

# Приводим к виду [[], [], []]. Пример: было [1, 2, 3] стало [[1], [2], [3]]
x = ds[X_NAME].values.reshape(-1, 1)
y = ds[Y_NAME].values.reshape(-1, 1)

# Нормализация данных
# y = normalize(y, axis=0).ravel()

# Рисуем по сырым данным
draw_graph(ds)

# Проверка на нормальность
# statistic, pvalue = shapiro(y)
# print(statistic, pvalue)

# Разделение данных
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Обучение модели
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Коэф. регрессии
print(regressor.intercept_, regressor.coef_)

# Построение прогноза
y_pred = regressor.predict(x_test)
df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})

# Посмотреть на справнение прогноза и реальных данных
# df1 = df.head(10)
# df1.plot(kind="bar")
# plt.show()

draw_reg(x_test, y_test, y_pred, "green")
draw_reg(scale_array(x_test),scale_array(y_test), scale_array(y_pred), "red")
