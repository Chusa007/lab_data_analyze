# -*- coding: utf-8 -*-

import math
import random

def linear_regression(a: int=0, b: int=0, x: int=0):
    """
    Формула линейнй регрессии в общем виде y=a+b*x
    """
    y = a + b * x
    print("y = {y}".format(y=y))


def polynomial_regression(a: int=0, x: int=0, power: int=0):
    """
    Формула нелинейной полиномиальной регрессии в общем виде y=a+b1*x+b2*x^2+b3*x^3....
    """
    if power == 0:
        print("y = {a}".format(a=a))
        return

    b_coef = [random.randint(0, 10) for i in range(power)]
    print("coef = {coef}".format(coef=b_coef))
    sum_b = 0
    for b in b_coef:
        sum_b += b * math.pow(x, b_coef.index(b) + 1)
    print("sum_b = {sum_b}".format(sum_b=sum_b))
    y = a + sum_b
    print("y = {y}".format(y=y))


def giperbol_regression(a: int=0, b: int=0, x: int=0):
    """
    Формула нелинейной гиперболической регрессии в общем виде y=a+b/x
    """
    if x == 0:
        print("Неверный данные")

    y = a + b / x
    print("y = {y}".format(y=round(y, 3)))


def power_regression(a: int=0, b: int=0, x: int=0):
    """
    Формула нелинейной степенной регрессии в общем виде y=a+x^b
    """
    y = a + math.pow(x, b)
    print("y = {y}".format(y=y))


def pokaz_regression(a: int=0, b: int=0, x: int=0):
    """
    Формула нелинейной показательной регрессии в общем виде y=a+x^b
    """
    y = a + math.pow(b, x)
    print("y = {y}".format(y=y))


def exponential_regression(b0: int = 0, b1: int = 0, x: int = 0):
    """
    Формула нелинейной экспоненциальная регрессии в общем виде y=a+x^b
    """
    y = b0 * math.exp(b1*x)
    print("y = {y}".format(y=y))


def mathematical_expectation(count: int):
    """
    Формула математического ожидания дискретной случайной величины E(x) = sum(Xi*Bi)
    """
    b_coef = [random.randint(0, 10) for i in range(count)]
    x_coef = [random.randint(0, 10) for i in range(count)]
    bx_coef = [bc * xc for bc, xc in zip(b_coef, x_coef)]
    e = sum(bx_coef)

    print("b_coef = {b_coef}".format(b_coef=b_coef))
    print("x_coef = {x_coef}".format(x_coef=x_coef))
    print("bx_coef = {bx_coef}".format(bx_coef=bx_coef))
    print("e = {e}".format(e=e))


def dispers_random_value(count: int):
    """
    Дисперсия случайной величины Х, заданная дискретным рядом распределения
    D(X) = M(X^2) - (M(X))^2
    """
    x_coef = [random.randint(0, 10) for i in range(count)]
    p_coef = [round(random.random(), 2) for i in range(count)]

    mx = sum([pc * xc for pc, xc in zip(x_coef, p_coef)])
    m_xpow = sum([math.pow(xc, 2) * pc for xc, pc in zip(x_coef, p_coef)])
    d = m_xpow - math.pow(mx, 2)

    print("mx = {mx}".format(mx=mx))
    print("m_xpow = {m_xpow}".format(m_xpow=m_xpow))
    print("x_coef = {x_coef}".format(x_coef=x_coef))
    print("p_coef = {p_coef}".format(p_coef=p_coef))
    print("d = {d}".format(d=d))


def normal_distribution():
    """
    Формула функции плотности нормального распределени (1/b*sqrt(2*p))*e^(-pow(x-m, 2)/2*pow(b, 2))
    где x - значение изменяющейся величины, m - среднее значение, b - стандартное отклонение,
    e=2,71828... - основание натурального логарифма, p =3,1416...
    """
    b = round(random.random(), 3)
    m = round(random.random(), 3)
    x = random.randint(1, 20)

    fraction = 1 / b * math.sqrt(2 * math.pi)
    exponent = math.exp(math.pow(x - m, 2) * -1 / (2 * math.pow(b, 2)))
    fx = fraction * exponent

    print("b = {b}".format(b=b))
    print("m = {m}".format(m=m))
    print("x = {x}".format(x=x))
    print("fraction = {fraction}".format(fraction=fraction))
    print("exponent = {exponent}".format(exponent=exponent))
    print("fx = {fx}".format(fx=fx))


# linear_regression(3, 2, 3)
# polynomial_regression(3, 2, 3)
# giperbol_regression(3, 2, 3)
# power_regression(3, 2, 3)
# pokaz_regression(3, 2, 3)
# exponential_regression(3, 2, 3)
# mathematical_expectation(4)
# dispers_random_value(4)
# normal_distribution()
