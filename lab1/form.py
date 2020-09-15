# -*- coding: utf-8 -*-

import math
import random

def linear_regression(a: int=0, b: int=0, x: int=0):
    """
    Формула линейнй регрессии в общем виде y=a+b*x
    """
    y = a + b * x
    print(y)


def polynomial_regression(a: int=0, x: int=0, power: int=0):
    """
    Формула нелинейной полиномиальной регрессии в общем виде y=a+b1*x+b2*x^2+b3*x^3....
    """
    if power == 0:
        print(a)
        return

    b_coef = [random.randint(0, 10) for i in range(power)]
    print("coef = {coef}".format(coef=b_coef))
    sum_b = 0
    for b in b_coef:
        sum_b += b * math.pow(x, b_coef.index(b) + 1)
    print("sum_b = {sum_b}".format(sum_b=sum_b))
    y = a + sum_b
    print(y)


def giperbol_regression(a: int=0, b: int=0, x: int=0):
    """
    Формула нелинейной гиперболической регрессии в общем виде y=a+b/x
    """
    if x == 0:
        print("Неверный данные")

    y = a + b / x
    print(round(y, 3))


def power_regression(a: int=0, b: int=0, x: int=0):
    """
    Формула нелинейной степенной регрессии в общем виде y=a+x^b
    """
    y = a + math.pow(x, b)
    print(y)


def pokaz_regression(a: int=0, b: int=0, x: int=0):
    """
    Формула нелинейной показательной регрессии в общем виде y=a+x^b
    """
    y = a + math.pow(b, x)
    print(y)


def exponential_regression(b0: int = 0, b1: int = 0, x: int = 0):
    """
    Формула нелинейной экспоненциальная регрессии в общем виде y=a+x^b
    """
    y = b0 * math.exp(b1*x)
    print(y)


# linear_regression(3, 2, 3)
# polynomial_regression(3, 2, 3)
# giperbol_regression(3, 2, 3)
# power_regression(3, 2, 3)
# pokaz_regression(3, 2, 3)
exponential_regression(3, 2, 3)