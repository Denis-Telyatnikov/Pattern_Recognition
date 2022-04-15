import numpy as np
import matplotlib.pyplot as plt
import math
import pickle


k = 12
L = 0

file = open("sample.bin", "rb")
x = pickle.load(file)
y = pickle.load(file)
c_x = pickle.load(file)
c_y = pickle.load(file)
x1 = pickle.load(file)
y1 = pickle.load(file)
file.close()

X = np.array([[a ** n for n in range(k + 1)] for a in x])
IL = np.array([[L if i == j else 0 for j in range(k + 1)] for i in range(k + 1)])
IL[0][0] = 0

A = np.linalg.inv(X.T @ X + IL)
w = (A @ X.T) @ y
# print(w)


y2 = []
for t in x1:
    r = 0
    for j in range(k + 1):
        r += w[j] * (t ** j)
    y2.append(r)


plt.scatter(x, y, s=20, c='g')
plt.plot(x1, y1, linewidth=1)
plt.plot(x1, y2, linewidth=2)

plt.grid(True)
plt.show()
