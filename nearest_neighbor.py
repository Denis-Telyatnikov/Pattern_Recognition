import random


# Вспомогательные функции
def show_binary(matrix):
    for i in matrix:
        print(*i, sep='')


def show_mono(matrix):
    for i in matrix:
        for j in i:
            if j < 10:
                print(j, "   ", end='')
            elif 10 <= j < 100:
                print(j, '  ', end='')
            elif 100 <= j:
                print(j, ' ', end='')
        print()


def show_standard(digit):
    s = str(digit) + '.txt'
    f = open(s)
    for line in f:
        print(line, end='')
    print()
    f.close()


# Из файла-эталона в матрицу целых чисел с заданными цветами
def convert_to_matrix(digit, white, black):
    matrix = []
    s = str(digit) + '.txt'
    f = open(s)
    for line in f:
        lst = list(line)
        lst2 = []
        for t in lst:
            if t == "0":
                lst2.append(white)
            elif t == "1":
                lst2.append(black)
        matrix.append(lst2)
    f.close()
    return matrix


# Из матрицы в список
def make_list(matrix):
    lst2 = []
    for lst in matrix:
        for i in lst:
            lst2.append(i)
    return lst2


# Эталон как одномерный список с заданными цветами
def get_standard(digit, white, black):
    s = str(digit) + '.txt'
    f = open(s)
    standard = []
    for line in f:
        lst = list(line)
        for i in lst:
            if i == '0':
                standard.append(white)
            if i == '1':
                standard.append(black)
    return standard


# Добавление шума для бинарных изображений
def add_noise_binary(digit, chance):
    matrix = []
    s = str(digit) + '.txt'
    f = open(s)
    for line in f:
        lst = list(line)
        lst2 = []
        for t in lst:
            if t == "0":
                data = random.choices([0, 1], weights=[1 - chance, chance], k=1)[0]
                lst2.append(data)
            elif t == "1":
                data = random.choices([0, 1], weights=[chance, 1 - chance], k=1)[0]
                lst2.append(data)
        matrix.append(lst2)
    f.close()
    return matrix


def add_noise_single(digit, chance):
    matrix = []
    s = str(digit) + '.txt'
    f = open(s)
    for line in f:
        lst = list(line)
        lst2 = []
        for t in lst:
            if t == "0":
                data = random.choices([0, 1], weights=[1 - chance, chance], k=1)[0]
                lst2.append(data)
            elif t == "1":
                lst2.append(1)
        matrix.append(lst2)
    f.close()
    return matrix


# Метрики
# Квадрат евклидова расстояния
def square_euclidean_distance(lst_noise, white, black):
    list_dist = []
    dist = 0
    for i in range(10):

        lst2 = get_standard(i, white, black)
        for t in range(len(lst_noise)):
            dist += (lst2[t] - lst_noise[t]) ** 2
        list_dist.append(dist)
        dist = 0
    return list_dist


# Манхэттенское расстояни
def manhattan_distance(lst_noise, white, black):
    list_dist = []
    dist = 0
    for i in range(10):
        lst2 = get_standard(i, white, black)
        for t in range(len(lst_noise)):
            dist += abs(lst2[t] - lst_noise[t])
        list_dist.append(dist)
        dist = 0
    return list_dist


# Расстояние Чебышёва
def chebyshev_distance(lst_noise, white, black):
    list_dist = []
    for i in range(10):
        lst2 = get_standard(i, white, black)
        interim_list = [0]
        for t in range(len(lst_noise)):
            dist = abs(lst2[t] - lst_noise[t])
            if dist not in interim_list:
                interim_list.append(dist)
        list_dist.append(max(interim_list))
    return (list_dist)


# Добавление шума для цифр в градациях серого
def add_noise1(matrix, chance, w, lw, rw, b, lb, rb):
    matrix2 = []
    lstw = [i for i in range(0, lw)]
    lstw.extend([i for i in range(lw + 1, lw + rw + 1)])
    lstb = [i for i in range(0, lb)]
    lstb.extend([i for i in range(lb + 1, lb + rb + 1)])
    for lst in matrix:
        lst2 = []
        for i in lst:
            if i == w:
                n = (i - lw) + random.choice(lstw)
                data = random.choices([w, n], weights=[1 - chance, chance], k=1)[0]
                lst2.append(data)
            elif i == b:
                n = (i - lb) + random.choice(lstb)
                data = random.choices([b, n], weights=[1 - chance, chance], k=1)[0]
                lst2.append(data)
        matrix2.append(lst2)
    return matrix2


def add_noise2(matrix, w, lw, rw, b, lb, rb):
    w1 = w - lw
    b1 = b - lb
    matrix2 = []
    for lst in matrix:
        lst2 = []
        for i in lst:
            if i == w:
                data = w1 + random.randint(0, lw + rw)
                lst2.append(data)
            elif i == b:
                data = b1 + random.randint(0, lb + rb)
                lst2.append(data)
        matrix2.append(lst2)
    return matrix2


def add_noise3(matrix, chance, w, left, right):
    lstw = [i for i in range(0, left)]
    lstw.extend([i for i in range(left + 1, left + right + 1)])
    matrix2 = []
    for lst in matrix:
        lst2 = []
        for i in lst:
            if i == w:
                n = (i - left) + random.choice(lstw)
                data = random.choices([w, n], weights=[1 - chance, chance], k=1)[0]
                lst2.append(data)
            else:
                lst2.append(i)
        matrix2.append(lst2)
    return matrix2


def add_noise4(matrix, chance, w, lw, rw, b, lb, rb):
    matrix2 = []
    for lst in matrix:
        lst2 = []
        for i in lst:
            if i == w:
                n = random.randint(lw, rw)
                data = random.choices([w, n], weights=[1 - chance, chance], k=1)[0]
                lst2.append(data)
            elif i == b:
                n = random.randint(lb, rb)
                data = random.choices([b, n], weights=[1 - chance, chance], k=1)[0]
                lst2.append(data)
        matrix2.append(lst2)
    return matrix2
