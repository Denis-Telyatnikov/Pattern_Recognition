import numpy as np
import matplotlib.pyplot as plt
import math
import pickle


# ---   Нейронная сеть, состоящая
# ---   из 1 слоя с произвольным
# ---   числом нейронов


# np.random.seed(4)

# --- Функция активации - гиперболический тангенс
def activation_function(x):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)


def derivative_activation_function(t):
    return 1 - activation_function(t) ** 2


neurons = 20  # Количество нейронов в слое

iterations = 1000  # Количество итераций обучения (шагов ГС)

epochs = 5  # Количество эпох обучения

LR1 = 0.0001  # Скорость обучения

# Выборка
file = open("sample.bin", "rb")
x = pickle.load(file)
y = pickle.load(file)
c_x = pickle.load(file)
c_y = pickle.load(file)
x1 = pickle.load(file)
y1 = pickle.load(file)
file.close()

a, b = -2, 2

wI_m, wO_m, bs1_m, bsO_m = 0, 0, 0, 0
E_min = 10 ** 6
epoch_number = 0
for ep in range(epochs):
    LR = LR1
    wI, wO, bs1, bs2 = 0, 0, 0, 0
    E_m = 10 ** 6
    print(f'Эпоха {ep + 1}:')
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

        # if it == 1501:
        #     LR = LR / 2
        # elif it == 3001:
        #     LR = LR / 5

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

        print(f"Итерация {it}:", E)
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

y2 = []
for t in x1:
    a_1 = t * wI_m + bs1_m
    z_1 = activation_function(a_1)
    a_O = z_1.dot(wO_m) + bsO_m
    y2.append(a_O)




E_control = 0
for i in range(len(c_x)):
    a_1 = c_x[i] * wI_m + bs1_m
    z_1 = activation_function(a_1)
    a_O = z_1.dot(wO_m) + bsO_m
    E_control += 0.5 * (c_y[i] - a_O) ** 2
print("\nОшибка на контрольной выборке:", E_control)


plt.scatter(x, y, s=20, c='g')
plt.scatter(c_x, c_y, s=20, c='grey')
plt.plot(x1, y1, linewidth=1)
plt.plot(x1, y2, linewidth=2)
plt.grid(True)
plt.show()

