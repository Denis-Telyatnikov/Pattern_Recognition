import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- Валидация на отложенных данных (Hold-Out Validation)
# --- Обучающая выборка один раз случайным образом разбивается на две части,
# --- в данном случее это {x, y} (тренировочная выборка) и {c_x, c_y} (контрольная выборка)
# --- Выбирается модель с наименьшей ошибкой на контрольной выборке

# --- Метод Hold-out применяется в случаях больших датасетов,
# --- т.к. требует меньше вычислительных мощностей по сравнению
# --- с другими методами кросс-валидации. Недостатком метода
# --- является то, что оценка существенно зависит от разбиения,
# --- тогда как желательно, чтобы она характеризовала только алгоритм обучения.

k_up = 20
L_up = 0.5

# Выборка
file = open("sample.bin", "rb")
x = pickle.load(file)
y = pickle.load(file)
c_x = pickle.load(file)
c_y = pickle.load(file)
x1 = pickle.load(file)
y1 = pickle.load(file)
file.close()

E_min = 10**6
k_min = 0
L_min = 0

for k in range(k_up+1):
    h = 0.0001
    if k <= 10:
        lp = 0
    else:
        lp = L_up
    for L in np.arange(0, lp+h, h):
        print(f"Степень полинома - {k},  параметр регуляризации - {L}")
        E_cv = []
        X = np.array([[a ** n for n in range(k + 1)] for a in x])
        IL = np.array([[L if i == j else 0 for j in range(k + 1)] for i in range(k + 1)])
        IL[0][0] = 0
        A = np.linalg.inv(X.T @ X + IL)
        w = (A @ X.T) @ y
        # print(w)
        E_control = 0
        for t in range(len(c_x)):
            r = 0
            for j in range(k + 1):
                r += w[j] * (c_x[t] ** j)
            E_control += 0.5 * (c_y[t] - r) ** 2
        print(f"Ошибка на контрольной выборке: {E_control}")
        if E_control < E_min:
            E_min = E_control
            k_min = k
            L_min = L

print(f"Итог: степень полинома - {k_min},  параметр регуляризации - {L_min}")

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

plt.scatter(x, y, s=20, c='g')
plt.scatter(c_x, c_y, s=20, c='grey')
plt.plot(x1, y1, linewidth=1)
plt.plot(x1, y2, linewidth=2)

plt.grid(True)
plt.show()
