# -*- coding: utf-8 -*-

import numpy
from PIL import Image, ImageOps
from numpy import asarray

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import activations
from keras import optimizers
from keras.models import model_from_json
from keras.utils import np_utils

# Устанавливаем seed для повторяемости результатов
numpy.random.seed(42)
# Размер изображения
img_rows, img_cols = 28, 28


def create():
    # Загружаем данные
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Преобразование размерности изображений
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


    # Нормализация данных
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Преобразуем метки в категории
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    # Создаем последовательную модель
    model = Sequential()

    model.add(Conv2D(75, kernel_size=(2, 2),
                     activation=activations.tanh,
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(100, (2, 2), activation=activations.tanh))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation=activations.tanh))
    model.add(Dense(10, activation=activations.tanh))

    # Компилируем модель
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=["accuracy"])

    print(model.summary())

    # Обучаем сеть
    model.fit(X_train, Y_train, batch_size=200, epochs=30, validation_split=0.2, verbose=2)

    # Оцениваем качество обучения сети на тестовых данных
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

    print("Сохраняем сеть")
    # Сохраняем сеть для последующего использования
    # Генерируем описание модели в формате json
    model_json = model.to_json()
    json_file = open("adam_model30.json", "w")
    # Записываем архитектуру сети в файл
    json_file.write(model_json)
    json_file.close()
    # Записываем данные о весах в файл
    model.save_weights("adam_model30.h5")
    print("Сохранение сети завершено")


def test(name: str):
    print("Загружаю сеть из файлов")
    # Загружаем данные об архитектуре сети
    json_file = open("adam_model30.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    # Создаем модель
    loaded_model = model_from_json(loaded_model_json)
    # Загружаем сохраненные веса в модель
    loaded_model.load_weights("adam_model30.h5")
    print("Загрузка сети завершена")

    # Загрузка изображения
    image = Image.open("Image/"+name)
    # Ресайз изображения
    image = image.resize((img_rows, img_cols))
    # Конвертация в монохромный формат
    image = image.convert('1')
    # Конвертация в narray
    data = asarray(image)
    # Инвертация цветов монохромного
    data = numpy.logical_not(data)
    # Конвертация во входной тип
    data = data.astype('float32')
    data = numpy.array([data.reshape(img_rows, img_cols, 1)])
    # Запуск модели
    result = loaded_model.predict(data)
    # Вывод ответа
    result = result.tolist()[0]
    result = list(map(lambda x : [x, result.index(x)], result))
    result.sort(reverse=True)

    print(str(result))
    for i in range(3):
        print("С вероятностью " + str(int(result[i][0] * 100)) + "% на изображение цифра " + str(result[i][1]))

#
# test_name = ["test5.jpg", "test33.jpg", "test2.png", "test8.jpg"]
# for name in test_name:
#     test(name)
#  create()
