import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

# --- Случайные разбиения (Random subsampling)
# --- Исходная выборка {x, y} разбивается в случайной пропорции на две части:
# --- {xp, yp} - тренировочная выборка и {test_x, test_y} - тестовая выборка.
# --- Процедура повторяется несколько раз (параметр section).
# --- Выбирается модель с наименьшей средней ошибкой на полученных тестовых подвыборках.

k_up = 20
L_up = 0.01

section = 5

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
    h = 0.00001
    if k <= 3:
        lp = 0
    else:
        lp = L_up
    for L in np.arange(0, lp+h, h):
        print(f"Степень полинома - {k},  параметр регуляризации - {L}")
        E_cv = []
        for c in range(section):
            proportion = random.randint(15, 50)
            number = round(len(x) * (proportion / 100))
            t = np.random.choice(range(len(x)), size=number, replace=False)
            test_x, test_y = [], []
            for q in t:
                test_x.append(x[q])
                test_y.append(y[q])
            xp, yp = [], []
            for z in range(len(x)):
                if x[z] not in test_x:
                    xp.append(x[z])
                    yp.append(y[z])
            # print(f"-{c + 1}- размер тренировочной выборки - {len(xp)}, "
            #       f"размер тестовой выборки - {len(test_x)}")
            X = np.array([[a ** n for n in range(k + 1)] for a in xp])
            IL = np.array([[L if i == j else 0 for j in range(k + 1)] for i in range(k + 1)])
            IL[0][0] = 0

            A = np.linalg.inv(X.T @ X + IL)
            w = (A @ X.T) @ yp
            # print(w)

            E_test = 0
            for t in range(len(test_x)):
                r = 0
                for j in range(k + 1):
                    r += w[j] * (test_x[t] ** j)
                E_test += 0.5 * (test_y[t] - r) ** 2
            E_cv.append(E_test)
        E_average = (1/section)*sum(E_cv)
        print(f"Средняя ошибка валидации: {E_average}")
        if E_average < E_min:
            E_min = E_average
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