import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

np.random.seed(10)

N = 200  # размер обучающей выборки
control = 50  # размер контрольной выборки

file = open("sample.bin", "wb")

# ----cos(0.4pix)----
# s1, s2 = -2*math.pi, 2*math.pi
# sample = np.array([np.random.uniform(s1, s2) for _ in range(N+control)])
# # контрольная выборка
# c_x = np.random.choice(sample, size=control, replace=False)
# c_y = np.cos(0.4 * math.pi * c_x) + np.random.normal(0, 0.3, control)
# # обучающая выборка
# x = np.setdiff1d(sample, c_x)
# y = np.cos(0.4 * math.pi * x) + np.random.normal(0, 0.3, N)
# # исходная функциональная зависимость
# x1 = np.array(sorted([np.random.uniform(s1, s2) for _ in range(500)]))
# y1 = np.cos(0.4 * math.pi * x1)
# ------------------


# ----5*x^3+x^2+5----
s1, s2 = -1.5, 1.5
sample = np.array([np.random.uniform(s1, s2) for _ in range(N+control)])
# контрольная выборка
c_x = np.random.choice(sample, size=control, replace=False)
c_y = (5 * (c_x ** 3) + (c_x ** 2) + 5) + np.random.normal(0, 1.5, control)
# обучающая выборка
x = np.setdiff1d(sample, c_x)
y = (5 * (x ** 3) + (x ** 2) + 5) + np.random.normal(0, 1.5, N)
# исходная функциональная зависимость
x1 = np.array(sorted([np.random.uniform(s1, s2) for _ in range(500)]))
y1 = (5 * (x1 ** 3) + (x1 ** 2) + 5)
# ------------------


# ----x*sin(1.7pi*x)----
# s1, s2 = -2, 2
# sample = np.array([np.random.uniform(s1, s2) for _ in range(N+control)])
# # контрольная выборка
# c_x = np.random.choice(sample, size=control, replace=False)
# c_y = c_x*np.sin(1.7 * math.pi * c_x) + np.random.normal(0, 0.1, control)
# # обучающая выборка
# x = np.setdiff1d(sample, c_x)
# y = x*np.sin(1.7 * math.pi * x) + np.random.normal(0, 0.1, N)
# # исходная функциональная зависимость
# x1 = np.array(sorted([np.random.uniform(s1, s2) for _ in range(500)]))
# y1 = x1*np.sin(1.7 * math.pi * x1)
# ------------------


# ----3*sin(0.3 * pi * x)----
# s1, s2 = -7, 7
# sample = np.array([np.random.uniform(s1, s2) for _ in range(N+control)])
# # контрольная выборка
# c_x = np.random.choice(sample, size=control, replace=False)
# c_y = 3*np.sin(0.3 * math.pi * c_x) + np.random.normal(0, 0.4, control)
# # обучающая выборка
# x = np.setdiff1d(sample, c_x)
# y = 3*np.sin(0.3 * math.pi * x) + np.random.normal(0, 0.4, N)
# # исходная функциональная зависимость
# x1 = np.array(sorted([np.random.uniform(s1, s2) for _ in range(500)]))
# y1 = 3*np.sin(0.3 * math.pi * x1)
# ------------------


# обучающая выборка
pickle.dump(x, file)
pickle.dump(y, file)

# контрольная выборка
pickle.dump(c_x, file)
pickle.dump(c_y, file)

# исходная функциональная зависимость
pickle.dump(x1, file)
pickle.dump(y1, file)

file.close()

plt.scatter(x, y, s=20, c='g')
plt.scatter(c_x, c_y, s=20, c='orange')
plt.plot(x1, y1, linewidth=1)

# plt.suptitle('5*x^3+x^2+5')

plt.grid(True)
plt.show()

