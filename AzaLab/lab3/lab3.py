# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras import activations
from keras import optimizers


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


def count_errors(y_test, y_pred) -> dict:
    """
    Подсчет ошибок 1 и 2 рода
    :param y_test - тестовые данные
    :param y_pred - предсказанные данные
    :return: возвращает количество ошибок 1 и 2 рода
    """
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

# Чтение данных из файла
data_set = pd.read_csv("Development Index.csv")
data_set.sample(frac=1)
y = np.array(data_set.pop("Development Index"))

# Обучение модели классификации
# Разделение данных
x_train, x_test, y_train, y_test = train_test_split(data_set, y, stratify=y, test_size=0.33, random_state=0)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

# Нормализация обучающей выборки
scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))
scalerX = scalerX.fit(data_set.to_numpy())
scalerY = scalerY.fit(y.reshape(len(y), 1))

x_train_scale = scalerX.transform(x_train)
x_test_scale = scalerX.transform(x_test)
y_train_scale = scalerY.transform(y_train.reshape(len(y_train), 1))

model = Sequential()
# Входной слой
model.add(Input(shape=(6)))
# 1 слой
model.add(Dense(10, activation=activations.tanh, activity_regularizer='l1'))
model.add(Dropout(0.2))
# 2 слой
model.add(Dense(10, activation=activations.tanh, activity_regularizer='l1'))
model.add(Dropout(0.2))
# Выходной слой
model.add(Dense(1, activation=activations.tanh, activity_regularizer='l1'))

# Компилирование модели
# default value learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
              metrics=['accuracy'])
# Обучение модели
model.fit(x=x_train_scale, y=y_train_scale, epochs=200, batch_size=10)
scores = model.evaluate(x_train_scale, y_train_scale)

# Построение прогноза
y_pred_scale = model.predict(x_test_scale)
y_pred = scalerY.inverse_transform(y_pred_scale)
y_pred = np.rint(y_pred)

if SHOW_GRAPH:
    # Сравнение прогноза с реальными значениями
    df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
    # Посмотреть на справнение прогноза и реальных данных
    df1 = df.head(20)
    df1.plot(kind="bar")
    plt.show()

# Вычисление метрик
f1_res = f1_score(y_test, y_pred, average='micro')

print(count_errors(y_test, y_pred))
print("f1 = {f1}".format(f1=f1_res))
