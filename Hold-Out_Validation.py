import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import math
import pickle

# --- В данном случае контрольная выборка используется для построения
# --- различных зависимостей ошибки от параметров модели


# --- Функция активации - гиперболический тангенс
def activation_function(x):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)


def derivative_activation_function(t):
    return 1 - activation_function(t) ** 2


# Выборка
file = open("sample.bin", "rb")
x = pickle.load(file)
y = pickle.load(file)
c_x = pickle.load(file)
c_y = pickle.load(file)
x1 = pickle.load(file)
y1 = pickle.load(file)
file.close()


# --- Построение зависимости ошибки на контрольной выборке от
# --- количества слоев и количества нейронов в каждом слое.
# --- layers_up - максимальное количество слоев, для которых строится зависимость.
# --- neurons_up - максимальное количество нейронов в каждом слое.
# --- epochs - количество эпох обучения; iterations - количество итераций обучения (шагов ГС).
# --- LR - скорость обучения (размер шага ГС).
# --- {x, y} - обучающая выборка; {c_x, c_y} - контрольная выборка.
def show_error_dependencies_for_layers(layers_up, neurons_up, iterations, epochs,
                                       LR, x, y, c_x, c_y):
    E_l = []
    for layers in range(1, layers_up + 1):
        E_n = []
        for neurons in range(1, neurons_up + 1):
            if layers == 1:
                print(f"Количество слоев: {layers}, количество нейронов: {neurons}")
                a, b = -2, 2
                wI_m, wO_m, bs1_m, bsO_m = 0, 0, 0, 0
                E_min = 10 ** 6
                epoch_number = 0
                for ep in range(epochs):
                    wI, wO, bs1, bs2 = 0, 0, 0, 0
                    E_m = 10 ** 6
                    print(f'Эпоха {ep + 1}:', end=" ")
                    # Генерация начальных значений весов и смещений случайным образом
                    W_I = np.random.uniform(a, b, size=neurons)
                    b_1 = np.random.uniform(-5, 5, size=neurons)
                    W_O = np.random.uniform(a, b, size=neurons)
                    b_O = np.random.uniform(-5, 5)

                    for it in range(iterations + 1):
                        # -----Проход по первому элементу тренировочной выборки----
                        a1 = x[0] * W_I + b_1
                        z1 = activation_function(a1)
                        a_O = z1.dot(W_O) + b_O
                        E = 0.5 * (y[0] - a_O) ** 2
                        dE_daO = a_O - y[0]
                        dE_dbO = dE_daO
                        dE_dwO = z1 * dE_daO
                        dE_dz1 = W_O * dE_daO
                        dE_da1 = dE_dz1 * derivative_activation_function(a1)
                        dE_dwI = dE_da1 * x[0]
                        dE_db1 = dE_da1
                        for j in range(1, len(x)):
                            # ------------------------Прямое распространение--------------------
                            a1 = x[j] * W_I + b_1
                            z1 = activation_function(a1)
                            a_O = z1.dot(W_O) + b_O
                            E += 0.5 * (y[j] - a_O) ** 2
                            # ------------------------Обратное распространение-------------------
                            dE_daO = a_O - y[j]
                            dE_dbO += dE_daO
                            dE_dwO += z1 * dE_daO
                            dE_dz1 = W_O * dE_daO
                            dE_da1 = dE_dz1 * derivative_activation_function(a1)
                            dE_dwI += dE_da1 * x[j]
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
                    print("Ошибка:", E_m)
                    if E_m < E_min:
                        epoch_number = ep
                        E_min = E_m
                        wI_m, wO_m, bs1_m, bsO_m = wI, wO, bs1, bs2
                print(f"Наименьшая ошибка обучения (эпоха {epoch_number + 1}):", E_min)

                E_control = 0
                for i in range(len(c_x)):
                    a_1 = c_x[i] * wI_m + bs1_m
                    z_1 = activation_function(a_1)
                    a_O = z_1.dot(wO_m) + bsO_m
                    E_control += 0.5 * (c_y[i] - a_O) ** 2
                print("Ошибка на контрольной выборке:", E_control)
                print("----------------")
                E_n.append(E_control)

            else:
                print(f"Количество слоев: {layers}, количество нейронов: {neurons}")
                wI_m, wW_m, wO_m, bs1_m, bs2_m, bs3_m = 0, 0, 0, 0, 0, 0
                E_min = 10 ** 6
                epoch_number = 0
                for ep in range(epochs):
                    wI, wW, wO, bs1, bs2, bs3 = 0, 0, 0, 0, 0, 0
                    E_m = 10 ** 6
                    print(f'Эпоха {ep + 1}:', end=" ")

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
                        a_1 = x[0] * w_I + b_1
                        ak = a_1
                        for l in range(layers - 1):
                            z[l] = activation_function(ak)
                            a[l] = z[l].dot(W[l]) + b[l]
                            ak = a[l]
                        z_O = activation_function(a[layers - 2])
                        a_O = z_O.dot(w_O) + b_O
                        E = 0.5 * (y[0] - a_O) ** 2
                        dE_dW = [0 for _ in range(layers - 1)]
                        dE_db = [0 for _ in range(layers - 1)]
                        dE_dz = [0 for _ in range(layers - 1)]
                        dE_da = [0 for _ in range(layers - 1)]
                        dE_daO = a_O - y[0]
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
                        dE_dwI = dE_da1 * x[0]
                        dE_db1 = dE_da1

                        # -----Проход по остальным элементам тренировочной выборки----
                        for j in range(1, len(x)):
                            # --------------------Прямое распространение-----------------------
                            z = np.zeros((layers - 1, neurons))
                            a = np.zeros((layers - 1, neurons))
                            a_1 = x[j] * w_I + b_1
                            ak = a_1
                            for l in range(layers - 1):
                                z[l] = activation_function(ak)
                                a[l] = z[l].dot(W[l]) + b[l]
                                ak = a[l]
                            z_O = activation_function(a[layers - 2])
                            a_O = z_O.dot(w_O) + b_O
                            E += 0.5 * (y[j] - a_O) ** 2

                            # ---------------------Обратное распространение---------------------
                            dE_dW = [0 for _ in range(layers - 1)]
                            dE_db = [0 for _ in range(layers - 1)]
                            dE_dz = [0 for _ in range(layers - 1)]
                            dE_da = [0 for _ in range(layers - 1)]
                            dE_daO = a_O - y[j]
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
                            dE_dwI += dE_da1 * x[j]
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

                    print("Ошибка:", E_m)
                    if E_m < E_min:
                        epoch_number = ep
                        E_min = E_m
                        wI_m, wW_m, wO_m, bs1_m, bs2_m, bs3_m = wI, wW, wO, bs1, bs2, bs3

                print(f"Наименьшая ошибка обучения (эпоха {epoch_number + 1}):", E_min)

                E_control = 0
                for i in range(len(c_x)):
                    z = np.zeros((layers - 1, neurons))
                    a = np.zeros((layers - 1, neurons))

                    a_1 = c_x[i] * wI_m + bs1_m
                    ak = a_1
                    for l in range(layers - 1):
                        z[l] = activation_function(ak)
                        a[l] = z[l].dot(wW_m[l]) + bs2_m[l]
                        ak = a[l]
                    z_O = activation_function(a[layers - 2])
                    a_O = z_O.dot(wO_m) + bs3_m
                    E_control += 0.5 * (c_y[i] - a_O) ** 2
                print("Ошибка на контрольной выборке:", E_control)
                print("----------------")
                E_n.append(E_control)

        E_l.append(E_n)

    print(E_l)

    fig = plt.figure()
    ax = fig.add_subplot()
    fig.suptitle('Hold-Out Validation')
    plt.figtext(0.135, 0.135, f"Кол-во эпох: {epochs}\n"
    f"Кол-во итераций: {iterations}\nСкорость обучения: {LR}", fontsize=9)

    for i in range(len(E_l)):
        ax.plot(range(1, neurons_up + 1), E_l[i], label=f"layers - {i + 1}")

    ax.legend()
    ax.set_xlabel("neurons")
    ax.set_ylabel("E_control")
    # ax.xaxis.set_major_locator(MultipleLocator(base=1))
    plt.grid(True)
    plt.show()

    e = np.min(E_l)
    i, j = np.where(np.isclose(E_l, e))
    l_min = int(i) + 1
    n_min = int(j) + 1
    print('----------------')
    print(f"Наименьшая ошибка на контрольной выборке: {e}\n"
          f"Количество слоев: {l_min},   "
          f"Количество нейронов: {n_min}")


# --- Построение зависимости ошибки на контрольной выборке от
# --- количества нейронов (для НС из одного скрытого слоя).
# --- neurons_up - максимальное количество нейронов в слое.
# --- epochs - количество эпох обучения; iterations - количество итераций обучения (шагов ГС).
# --- LR1 - начальная скорость обучения (размер шага ГС), изменяется
# --- после определенного числа итераций обучения.
# --- {x, y} - обучающая выборка; {c_x, c_y} - контрольная выборка.
def show_dependence_error_on_neurons_for_one_layer(neurons_up, iterations, epochs,
                                                   LR1, x, y, c_x, c_y):
    E_n = []
    for neurons in range(1, neurons_up + 1):
        print(f'Количество нейронов: {neurons}')
        a, b = -2, 2
        wI_m, wO_m, bs1_m, bsO_m = 0, 0, 0, 0
        E_min = 10 ** 6
        epoch_number = 0
        for ep in range(epochs):
            LR = LR1
            wI, wO, bs1, bs2 = 0, 0, 0, 0
            E_m = 10 ** 6
            # print(f'Эпоха {ep + 1}:')
            # Генерация начальных значений весов и смещений случайным образом
            W_I = np.random.uniform(a, b, size=neurons)
            b_1 = np.random.uniform(-5, 5, size=neurons)
            W_O = np.random.uniform(a, b, size=neurons)
            b_O = np.random.uniform(-5, 5)

            for it in range(iterations + 1):
                # -----Проход по первому элементу тренировочной выборки----
                a1 = x[0] * W_I + b_1
                z1 = activation_function(a1)
                a_O = z1.dot(W_O) + b_O
                E = 0.5 * (y[0] - a_O) ** 2
                dE_daO = a_O - y[0]
                dE_dbO = dE_daO
                dE_dwO = z1 * dE_daO
                dE_dz1 = W_O * dE_daO
                dE_da1 = dE_dz1 * derivative_activation_function(a1)
                dE_dwI = dE_da1 * x[0]
                dE_db1 = dE_da1

                # ---------Изменение шага ГС после определенного числа итераций-----------
                if it == 2001:
                    LR = LR / 2
                elif it == 3001:
                    LR = LR / 5

                for j in range(1, len(x)):
                    # ------------------------Прямое распространение--------------------
                    a1 = x[j] * W_I + b_1
                    z1 = activation_function(a1)
                    a_O = z1.dot(W_O) + b_O
                    E += 0.5 * (y[j] - a_O) ** 2
                    # ------------------------Обратное распространение-------------------
                    dE_daO = a_O - y[j]
                    dE_dbO += dE_daO
                    dE_dwO += z1 * dE_daO
                    dE_dz1 = W_O * dE_daO
                    dE_da1 = dE_dz1 * derivative_activation_function(a1)
                    dE_dwI += dE_da1 * x[j]
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

        E_control = 0
        for i in range(len(c_x)):
            a_1 = c_x[i] * wI_m + bs1_m
            z_1 = activation_function(a_1)
            a_O = z_1.dot(wO_m) + bsO_m
            E_control += 0.5 * (c_y[i] - a_O) ** 2
        E_n.append(E_control)
        print(f"Ошибка на контрольной выборке:", E_control)
        print("----------\n")

    # print(E_n)

    fig = plt.figure()
    ax = fig.add_subplot()
    fig.suptitle('Hold-Out Validation')

    ax.plot(range(1, neurons_up + 1), E_n, label="1 скрытый слой")

    ax.legend()
    ax.set_xlabel("neurons")
    ax.set_ylabel("E_control")
    # ax.xaxis.set_major_locator(MultipleLocator(base=1))
    plt.grid(True)
    plt.show()

    e = min(E_n)
    n_min = E_n.index(e) + 1
    print('----------------')
    print(f"Наименьшая ошибка на контрольной выборке: {e}\n"
          f"Количество нейронов: {n_min}")


# show_error_dependencies_for_layers(layers_up=3, neurons_up=40,
#                                    iterations=1000, epochs=10,
#                                    LR=0.0001, x=x, y=y, c_x=c_x, c_y=c_y)


show_dependence_error_on_neurons_for_one_layer(neurons_up=105, iterations=5000,
                                               epochs=10, LR1=0.0001,
                                               x=x, y=y, c_x=c_x, c_y=c_y)
