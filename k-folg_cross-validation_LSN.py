import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import random

# --- k-fold кросс-валидация:
# --- 1. Обучающая выборка разбивается на k (в данном случае это параметр section)
# --- непересекающихся одинаковых по объему частей;
# --- 2. Производится k итераций. На каждой итерации происходит следующее:
# ---    а) Модель обучается на k−1 части обучающей выборки;
# ---    в) Модель тестируется на части обучающей выборки, которая не участвовала в обучении.
# --- Каждая из k частей единожды используется для тестирования.

# --- Выбирается модель с наименьшей средней ошибкой на тестовых подвыборках.

k_up = 20
L_up = 0.1

section = 20

# ---исходная выборка {x_i, f(x_i)} разбивается
# ---на подвыборки в случайном порядке.
def splitting_sample_1(x, y, parts):
    S = []
    for i in range(len(x)):
        S.append([x[i], y[i]])
    sx, sy = [], []
    sz = round(len(x) / parts)
    for _ in range(parts - 1):
        t = np.random.choice(range(len(S)), size=sz, replace=False)
        lst1, lst2 = [], []
        Sk = []
        for i in t:
            Sk.append(S[i])
            lst1.append(S[i][0])
            lst2.append(S[i][1])
        sx.append(lst1)
        sy.append(lst2)
        Lst = []
        for f in S:
            if f not in Sk:
                Lst.append(f)
        S = Lst
        # print(S)
    lst1, lst2 = [], []
    for f in S:
        lst1.append(f[0])
        lst2.append(f[1])
    sx.append(lst1)
    sy.append(lst2)
    return sx, sy


# ---блоки формируются исходя из разбиения
# ---интервала [min{x_i}, max{x_i}] на равные подинтервалы
def splitting_sample_2(x, y, parts):
    sx, sy = [], []
    mn, mx = np.min(x), np.max(x)
    h = (mx - mn) / parts
    n = mn
    for i in range(parts):
        lst1 = []
        lst2 = []
        for j in range(len(x)):
            if i == parts - 1:
                if n <= x[j] <= mx:
                    lst1.append(x[j])
                    lst2.append(y[j])
            else:
                if n <= x[j] < n + h:
                    lst1.append(x[j])
                    lst2.append(y[j])
        sx.append(lst1)
        sy.append(lst2)
        n += h
    return sx, sy

def creating_different_subsamples(xp, yp):
    matrix_x = []
    matrix_y = []
    for i in range(len(xp)):
        o_x = []
        o_y = []
        for j in range(len(xp)):
            if i == j:
                continue
            else:
                o_x.extend(xp[j])
                o_y.extend(yp[j])
        matrix_x.append(o_x)
        matrix_y.append(o_y)
    return matrix_x, matrix_y


file = open("sample.bin", "rb")
x = pickle.load(file)
y = pickle.load(file)
c_x = pickle.load(file)
c_y = pickle.load(file)
x1 = pickle.load(file)
y1 = pickle.load(file)
file.close()


xp, yp = splitting_sample_1(x, y, section)  # --1--
# xp, yp = splitting_sample_2(x, y, section)  # --2--

sb_x, sb_y = creating_different_subsamples(xp, yp)

# Как выглядит разбиение исходной выборки:
# for i in range(len(xp)):
#     plt.scatter(xp[i], yp[i], s=30)
# plt.plot(x1, y1, linewidth=1)
# plt.grid(True)
# plt.show()

E_min = 10**6
k_min, L_min = 0, 0

for k in range(k_up+1):
    h = 0.00001
    if k <= 1:
        lp = 0
    else:
        lp = L_up
    for L in np.arange(0, lp+h, h):
        print(f"Степень полинома - {k},  параметр регуляризации - {L}")
        E_cv = []
        for o in range(len(sb_x)):  # o - номер тестовой и тренировочной подвыборки
            X = np.array([[a ** n for n in range(k + 1)] for a in sb_x[o]])
            IL = np.array([[L if i == j else 0 for j in range(k + 1)] for i in range(k + 1)])
            IL[0][0] = 0

            A = np.linalg.inv(X.T @ X + IL)
            w = (A @ X.T) @ sb_y[o]
            # print(w)

            E_test = 0
            for t in range(len(xp[o])):
                r = 0
                for j in range(k + 1):
                    r += w[j] * (xp[o][t] ** j)
                E_test += 0.5 * (yp[o][t] - r) ** 2
            # print(f"Ошибка на тестовой выборке {o+1}:", E_test)
            # print("----------------")
            E_cv.append(E_test)
        E_average = (1/section)*sum(E_cv)
        print(f"Средняя ошибка валидации: {E_average}")
        if E_average < E_min:
            E_min = E_average
            k_min = k
            L_min = L

print(f"Параметры: степень полинома - {k_min},  параметр регуляризации - {L_min}")


X = np.array([[a ** n for n in range(k_min + 1)] for a in x])
IL = np.array([[L_min if i == j else 0 for j in range(k_min + 1)] for i in range(k_min + 1)])
IL[0][0] = 0
A = np.linalg.inv(X.T @ X + IL)
w = (A @ X.T) @ y

y2 = []
for t in x1:
    r = 0
    for j in range(k_min + 1):
        r += w[j] * (t ** j)
    y2.append(r)

E_control = 0
for t in range(len(c_x)):
    r = 0
    for j in range(k_min + 1):
        r += w[j] * (c_x[t] ** j)
    E_control += 0.5 * (c_y[t] - r) ** 2
print("----------------")
print(f"Ошибка на контрольной выборке:", E_control)

plt.scatter(x, y, s=20, c='g')
plt.scatter(c_x, c_y, s=20, c='grey')
plt.plot(x1, y1, linewidth=1)
plt.plot(x1, y2, linewidth=2)

plt.grid(True)
plt.show()
