import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import random

# Выборка
file = open("sample.bin", "rb")
x = pickle.load(file)
y = pickle.load(file)
c_x = pickle.load(file)
c_y = pickle.load(file)
x1 = pickle.load(file)
y1 = pickle.load(file)
file.close()


# ---Зависимость ошибки на контрольной выборке {c_x, c_y}
# ---от степени полинома (без регуляризации)
def get_dependence_accuracy_on_k(k_up, k_lower=0):
    L, k_min = 0, 0
    E_min = 10 ** 6
    E_k, w_min = [], []
    for k in range(k_lower, k_up + 1):
        print(f"Степень полинома - {k}")
        X = np.array([[a ** n for n in range(k + 1)] for a in x])
        IL = np.array([[L if i == j else 0 for j in range(k + 1)] for i in range(k + 1)])
        IL[0][0] = 0
        A = np.linalg.inv(X.T @ X + IL)
        w = (A @ X.T) @ y
        # print(w)
        E_test = 0
        for t in range(len(c_x)):
            r = 0
            for j in range(k + 1):
                r += w[j] * (c_x[t] ** j)
            E_test += 0.5 * (c_y[t] - r) ** 2
        E_k.append(E_test)
        print(f"Ошибка на отложенной выборке: {E_test}")
        if E_test < E_min:
            E_min = E_test
            k_min = k
            w_min = w

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot()
    fig.suptitle(f'Зависимость точности от степени полинома\n'
                 f'(min ошибка при k={k_min})', fontsize=10)
    ax.plot(range(k_lower, k_up + 1), E_k, linewidth=1)
    ax.scatter(k_min, E_min, s=30, c='r')
    ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel("E_control")
    plt.grid(True)
    plt.show()
    print(f"Степень полинома - {k_min} (наименьшая ошибка - {E_min})")
    return k_min, w_min


# ---Зависимость ошибки на контрольной выборке {c_x, c_y}
# ---от параметра регуляризации (фиксированная степень полинома)
def get_dependence_accuracy_on_L(k, L_up, h):
    L_min = 0
    E_min = 10 ** 6
    E_l, w_min = [], []
    print(f"Степень полинома - {k}")
    for L in np.arange(0, L_up + h, h):
        print(f"Параметр регуляризации - {L}")
        X = np.array([[a ** n for n in range(k + 1)] for a in x])
        IL = np.array([[L if i == j else 0 for j in range(k + 1)] for i in range(k + 1)])
        IL[0][0] = 0
        A = np.linalg.inv(X.T @ X + IL)
        w = (A @ X.T) @ y
        # print(w)
        E_test = 0
        for t in range(len(c_x)):
            r = 0
            for j in range(k + 1):
                r += w[j] * (c_x[t] ** j)
            E_test += 0.5 * (c_y[t] - r) ** 2
        E_l.append(E_test)
        print(f"Ошибка на отложенной выборке: {E_test}")
        if E_test < E_min:
            E_min = E_test
            L_min = L
            w_min = w

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot()
    fig.suptitle(f'Зависимость точности от параметра регуляризации\n'
                 f'Степень полинома {k}\n(min ошибка при L={L_min})', fontsize=10)
    ax.plot(np.arange(0, L_up + h, h), E_l, linewidth=1)
    ax.scatter(L_min, E_min, s=30, c='r')
    ax.legend()
    ax.set_xlabel("L")
    ax.set_ylabel("E_control")
    plt.grid(True)
    plt.show()
    print(f"Параметр регуляризации (cтепень {k}) - {L_min} (наименьшая ошибка - {E_min})")
    return k, w_min


# k, w = get_dependence_accuracy_on_k(27)  # --1--
k, w = get_dependence_accuracy_on_L(30, 0.5, 0.0001)  # --2--

y2 = []
for t in x1:
    r = 0
    for j in range(k + 1):
        r += w[j] * (t ** j)
    y2.append(r)

plt.scatter(x, y, s=20, c='g')
plt.scatter(c_x, c_y, s=20, c='grey')
plt.plot(x1, y1, linewidth=1)
plt.plot(x1, y2, linewidth=2)

plt.grid(True)
plt.show()
