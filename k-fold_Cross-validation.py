import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from idz.neural_network import start_network

# --- k-fold кросс-валидация:
# --- 1. Обучающая выборка разбивается на k (в данном случае это параметр section)
# --- непересекающихся одинаковых по объему частей;
# --- 2. Производится k итераций. На каждой итерации происходит следующее:
# ---    а) Модель обучается на k−1 части обучающей выборки;
# ---    в) Модель тестируется на части обучающей выборки, которая не участвовала в обучении.
# --- Каждая из k частей единожды используется для тестирования.

# --- Выбирается модель с наименьшей средней ошибкой на тестовых подвыборках.

layers_up = 3  # Количество слоев
neurons_up = 10  # Количество нейронов в слое
iterations = 1000  # Количество итераций обучения (шагов ГС)

epochs = 10  # Количество эпох обучения

LR = 0.0001  # Скорость обучения

section = 5


# --- Функция активации - гиперболический тангенс
def activation_function(x):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)


def derivative_activation_function(t):
    return 1 - activation_function(t) ** 2


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


# Выборка
file = open("sample.bin", "rb")
x = pickle.load(file)
y = pickle.load(file)
c_x = pickle.load(file)
c_y = pickle.load(file)
x1 = pickle.load(file)
y1 = pickle.load(file)
file.close()


# xp, yp = splitting_sample_1(x, y, section)  # --1--
xp, yp = splitting_sample_2(x, y, section)  # --2--

sb_x, sb_y = creating_different_subsamples(xp, yp)

E_l = []
for layers in range(1, layers_up + 1):
    E_n = []
    for neurons in range(1, neurons_up + 1):
        print(f"Количество слоев: {layers}, количество нейронов: {neurons}")
        E_cv = []
        for o in range(len(sb_x)):  # o - номер тестовой и тренировочной подвыборки
            if layers == 1:
                a, b = -2, 2
                wI_m, wO_m, bs1_m, bsO_m = 0, 0, 0, 0
                E_min = 10 ** 6
                epoch_number = 0
                for ep in range(epochs):
                    wI, wO, bs1, bs2 = 0, 0, 0, 0
                    E_m = 10 ** 6
                    # print(f'Эпоха {ep + 1}:', end=" ")
                    # Генерация начальных значений весов и смещений случайным образом
                    W_I = np.random.uniform(a, b, size=neurons)
                    b_1 = np.random.uniform(-5, 5, size=neurons)
                    W_O = np.random.uniform(a, b, size=neurons)
                    b_O = np.random.uniform(-5, 5)
                    for it in range(iterations + 1):
                        # -----Проход по первому элементу тренировочной выборки----
                        a1 = sb_x[o][0] * W_I + b_1
                        z1 = activation_function(a1)
                        a_O = z1.dot(W_O) + b_O
                        E = 0.5 * (sb_y[o][0] - a_O) ** 2
                        dE_daO = a_O - sb_y[o][0]
                        dE_dbO = dE_daO
                        dE_dwO = z1 * dE_daO
                        dE_dz1 = W_O * dE_daO
                        dE_da1 = dE_dz1 * derivative_activation_function(a1)
                        dE_dwI = dE_da1 * sb_x[o][0]
                        dE_db1 = dE_da1
                        for j in range(1, len(sb_x[o])):
                            # ------------------------Прямое распространение--------------------
                            a1 = sb_x[o][j] * W_I + b_1
                            z1 = activation_function(a1)
                            a_O = z1.dot(W_O) + b_O
                            E += 0.5 * (sb_y[o][j] - a_O) ** 2
                            # ------------------------Обратное распространение-------------------
                            dE_daO = a_O - sb_y[o][j]
                            dE_dbO += dE_daO
                            dE_dwO += z1 * dE_daO
                            dE_dz1 = W_O * dE_daO
                            dE_da1 = dE_dz1 * derivative_activation_function(a1)
                            dE_dwI += dE_da1 * sb_y[o][j]
                            dE_db1 += dE_da1
                        # print(f"Итерация {it}:", E)
                        # --------Обновление весов---------
                        W_I -= LR * dE_dwI
                        b_1 -= LR * dE_db1
                        W_O -= LR * dE_dwO
                        b_O -= LR * dE_dbO
                        if E < E_m:
                            E_m = E
                            wI = W_I
                            bs1 = b_1
                            wO = W_O
                            bs2 = b_O
                    # print("Ошибка:", E_m)
                    if E_m < E_min:
                        epoch_number = ep
                        E_min = E_m
                        wI_m, wO_m, bs1_m, bsO_m = wI, wO, bs1, bs2
                print(f"Наименьшая ошибка обучения (эпоха {epoch_number + 1}):", E_min)
                E_test = 0
                for i in range(len(yp[o])):
                    a_1 = xp[o][i] * wI_m + bs1_m
                    z_1 = activation_function(a_1)
                    a_O = z_1.dot(wO_m) + bsO_m
                    E_test += 0.5 * (yp[o][i] - a_O) ** 2
                print(f"Ошибка на тестовой подвыборке {o}:", E_test)
                # print("----------------")
                E_cv.append(E_test)

            else:
                wI_m, wW_m, wO_m, bs1_m, bs2_m, bs3_m = 0, 0, 0, 0, 0, 0
                E_min = 10 ** 6
                epoch_number = 0
                for ep in range(epochs):
                    wI, wW, wO, bs1, bs2, bs3 = 0, 0, 0, 0, 0, 0
                    E_m = 10 ** 6
                    # print(f'Эпоха {ep + 1}:', end=" ")

                    # Генерация начальных значений весов и смещений случайным образом
                    w_I = np.random.uniform(-2, 2, size=neurons)
                    b_1 = np.random.uniform(-5, 5, size=neurons)
                    W = np.random.uniform(-2, 2, (layers - 1, neurons, neurons))
                    b = np.random.uniform(-5, 5, (layers - 1, neurons))
                    w_O = np.random.uniform(-2, 2, size=neurons)
                    b_O = np.random.uniform(-5, 5)

                    for it in range(iterations + 1):
                        # -----Проход по первому элементу тренировочной выборки----
                        z = np.zeros((layers - 1, neurons))
                        a = np.zeros((layers - 1, neurons))
                        a_1 = sb_x[o][0] * w_I + b_1
                        ak = a_1
                        for l in range(layers - 1):
                            z[l] = activation_function(ak)
                            a[l] = z[l].dot(W[l]) + b[l]
                            ak = a[l]
                        z_O = activation_function(a[layers - 2])
                        a_O = z_O.dot(w_O) + b_O
                        E = 0.5 * (sb_y[o][0] - a_O) ** 2
                        dE_dW = [0 for _ in range(layers - 1)]
                        dE_db = [0 for _ in range(layers - 1)]
                        dE_dz = [0 for _ in range(layers - 1)]
                        dE_da = [0 for _ in range(layers - 1)]
                        dE_daO = a_O - sb_y[o][0]
                        dE_dbO = dE_daO
                        dE_dwO = z_O * dE_daO
                        dE_dzO = w_O * dE_daO
                        dzk = dE_dzO

                        for l in range(layers - 1):
                            dE_dal = dzk * derivative_activation_function(a[layers - 2 - l])
                            dE_da[layers - 2 - l] = dE_dal
                            dE_dWl = np.outer(z[layers - 2 - l], dE_dal)
                            dE_dW[layers - 2 - l] = dE_dWl
                            dE_dbl = dE_dal
                            dE_db[layers - 2 - l] = dE_dbl
                            dE_dzl = dE_dal.dot(W[layers - 2 - l].T)
                            dE_dz[layers - 2 - l] = dE_dzl
                            dzk = dE_dzl

                        dE_dW = np.array(dE_dW)
                        dE_db = np.array(dE_db)
                        dE_dz = np.array(dE_dz)
                        dE_da = np.array(dE_da)
                        dE_da1 = dE_dz[0] * derivative_activation_function(a_1)
                        dE_dwI = dE_da1 * sb_x[o][0]
                        dE_db1 = dE_da1

                        # -----Проход по остальным элементам тренировочной выборки----
                        for j in range(1, len(sb_x[o])):
                            # --------------------Прямое распространение-----------------------
                            z = np.zeros((layers - 1, neurons))
                            a = np.zeros((layers - 1, neurons))
                            a_1 = sb_x[o][j] * w_I + b_1
                            ak = a_1
                            for l in range(layers - 1):
                                z[l] = activation_function(ak)
                                a[l] = z[l].dot(W[l]) + b[l]
                                ak = a[l]
                            z_O = activation_function(a[layers - 2])
                            a_O = z_O.dot(w_O) + b_O
                            E += 0.5 * (sb_y[o][j] - a_O) ** 2

                            # ---------------------Обратное распространение---------------------
                            dE_dW = [0 for _ in range(layers - 1)]
                            dE_db = [0 for _ in range(layers - 1)]
                            dE_dz = [0 for _ in range(layers - 1)]
                            dE_da = [0 for _ in range(layers - 1)]
                            dE_daO = a_O - sb_y[o][j]
                            dE_dbO += dE_daO
                            dE_dwO += z_O * dE_daO
                            dE_dzO = w_O * dE_daO
                            dzk = dE_dzO

                            for l in range(layers - 1):
                                dE_dal = dzk * derivative_activation_function(a[layers - 2 - l])
                                dE_da[layers - 2 - l] = dE_dal
                                dE_dWl = np.outer(z[layers - 2 - l], dE_dal)
                                dE_dW[layers - 2 - l] += dE_dWl
                                dE_dbl = dE_dal
                                dE_db[layers - 2 - l] += dE_dbl
                                dE_dzl = dE_dal.dot(W[layers - 2 - l].T)
                                dE_dz[layers - 2 - l] = dE_dzl
                                dzk = dE_dzl

                            dE_dW += np.array(dE_dW)
                            dE_db += np.array(dE_db)
                            dE_dz = np.array(dE_dz)
                            dE_da = np.array(dE_da)
                            dE_da1 = dE_dz[0] * derivative_activation_function(a_1)
                            dE_dwI += dE_da1 * sb_x[o][j]
                            dE_db1 += dE_da1

                        # --------------------Обновление весов------------------
                        # print(f"Итерация {it}:", E)
                        w_I -= LR * dE_dwI
                        b_1 -= LR * dE_db1
                        for i in range(layers - 1):
                            W[i] -= LR * dE_dW[i]
                            b[i] -= LR * dE_db[i]
                        w_O -= LR * dE_dwO
                        b_O -= LR * dE_dbO

                        # if it > 0 and it % 10000:
                        #     LR -= 0.000005
                        if E < E_m:
                            E_m = E
                            wI, wW, wO = w_I, W, w_O
                            bs1, bs2, bs3 = b_1, b, b_O

                    # print("Ошибка:", E_m)
                    if E_m < E_min:
                        epoch_number = ep
                        E_min = E_m
                        wI_m, wW_m, wO_m, bs1_m, bs2_m, bs3_m = wI, wW, wO, bs1, bs2, bs3
                print(f"Наименьшая ошибка обучения (эпоха {epoch_number + 1}):", E_min)

                E_test = 0
                for i in range(len(yp[o])):
                    z = np.zeros((layers - 1, neurons))
                    a = np.zeros((layers - 1, neurons))

                    a_1 = xp[o][i] * wI_m + bs1_m
                    ak = a_1
                    for l in range(layers - 1):
                        z[l] = activation_function(ak)
                        a[l] = z[l].dot(wW_m[l]) + bs2_m[l]
                        ak = a[l]
                    z_O = activation_function(a[layers - 2])
                    a_O = z_O.dot(wO_m) + bs3_m
                    E_test += 0.5 * (yp[o][i] - a_O) ** 2
                print(f"Ошибка на тестовой подвыборке {o}:", E_test)
                # print("----------------")
                E_cv.append(E_test)

        E_average = (1 / section) * sum(E_cv)
        print(f"**Средняя ошибка валидации:", E_average)
        print("----------------")
        E_n.append(E_average)
    E_l.append(E_n)

print(E_l)
print("----------------")
e = np.min(E_l)
i, j = np.where(np.isclose(E_l, e))
l_min = int(i) + 1
n_min = int(j) + 1

print(f"Наименьшая средняя ошибка на тестовых подвыборках: {e}\n"
      f"Количество слоев: {l_min},   "
      f"Количество нейронов: {n_min}")

start_network(l_min, n_min, epochs, iterations, LR, x, y, c_x, c_y, x1, y1)
print("----------------")

E_l[int(i)][int(j)] = 10**6

e2 = np.min(E_l)
i2, j2 = np.where(np.isclose(E_l, e2))
l_min2 = int(i2) + 1
n_min2 = int(j2) + 1

print(f"Второй по точности результат: средняя ошибка на тестовых подвыборках - {e2}\n"
      f"Количество слоев: {l_min2},   "
      f"Количество нейронов: {n_min2}")

start_network(l_min2, n_min2, epochs, iterations, LR, x, y, c_x, c_y, x1, y1)
print("----------------")
