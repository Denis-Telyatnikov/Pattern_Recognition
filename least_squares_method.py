import numpy as np
import LSM.Gauss_method as gm
import matplotlib.pyplot as plt
import pickle


k = 8

file = open("sample.bin", "rb")
x = pickle.load(file)
y = pickle.load(file)
c_x = pickle.load(file)
c_y = pickle.load(file)
x1 = pickle.load(file)
y1 = pickle.load(file)
file.close()


def least_squares_method(k, x, y):
    a = [[0 for _ in range(k + 1)] for _ in range(k + 1)]
    b = [0 for _ in range(k + 1)]

    for i in range(len(x)):
        for p in range(k + 1):
            for q in range(k + 1):
                a[p][q] += x[i] ** (p + q)
            b[p] += (x[i] ** p) * y[i]

    coefficients = gm.Gauss(a, b)
    return coefficients


kf = least_squares_method(k, x, y)


y2 = []
for t in x1:
    r = 0
    for j in range(k + 1):
        r += kf[j] * (t ** j)
    y2.append(r)

plt.scatter(x, y, s=20, c='g')
plt.plot(x1, y1, linewidth=1)
plt.plot(x1, y2, linewidth=2)
plt.grid(True)
plt.show()
