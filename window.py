import tkinter as tk
import task_1.nearest_neighbor as nn

# --- Графический интерфейс
# --- Распознавание растровых изображений цифр с зашумлением
# --- с помощью метода ближайшего соседа


white = 0
black = 1

digit = 0
matrix = []
matrix_noise = []

runing = True


def run():
    def quit():
        window.destroy()

    def exit():
        global runing
        runing = False
        window.destroy()

    # Вывод незашумленных цифр
    def show_binary(matrix):
        text_b = ''
        for lst in matrix:
            for i in lst:
                text_b += str(i)
            text_b += '\n'
        text = text_b[0:-1]
        return text

    def show_mono(digit, w, b):
        global white, black, matrix
        white = int(w)
        black = int(b)
        matrix = nn.convert_to_matrix(digit, int(w), int(b))
        text2 = ''
        for lst in matrix:
            for i in lst:
                if i < 10:
                    text2 += str(i) + '   '
                elif 10 <= i < 100:
                    text2 += str(i) + '  '
                elif 100 <= i:
                    text2 += str(i) + ' '
            text2 += '\n'
        text = text2[0:-1]
        mess = tk.Message(window, text=text, width=1000, font=('Consolas', 10))
        mess.place(x=780, y=10)

    # Вывод зашумленных монохромных цифр
    def show_mono2(matrix):
        text2 = ''
        for lst in matrix:
            for i in lst:
                if i < 10:
                    text2 += str(i) + '   '
                elif 10 <= i < 100:
                    text2 += str(i) + '  '
                elif 100 <= i:
                    text2 += str(i) + ' '
            text2 += '\n'
        text = text2[0:-1]
        mess = tk.Message(window, text=text, width=1000, font=('Consolas', 10))
        mess.place(x=780, y=300)

    # Вывод зашумленных бинарных цифр
    def get_noise_binary(t, chanse):
        global digit, matrix_noise
        if t == 1:
            matrix_noise = nn.add_noise_binary(digit, float(chanse))
            text1 = show_binary(matrix_noise)
            mess = tk.Message(window, text=text1, width=1000, font=('Consolas', 11))
            mess.place(x=900, y=310)
        elif t == 2:
            matrix_noise = nn.add_noise_single(digit, float(chanse))
            text1 = show_binary(matrix_noise)
            mess = tk.Message(window, text=text1, width=1000, font=('Consolas', 11))
            mess.place(x=900, y=310)

    # Добавление шума в монохромные цифры
    def get_noise_monochrome(digit, t, lw, rw, lb, rb, chance=0.0):
        global matrix, matrix_noise
        if t == 2:
            matrix_noise = nn.add_noise2(matrix, white, int(lw), int(rw), black, int(lb), int(rb))
            show_mono2(matrix_noise)
        elif t == 1:
            matrix_noise = nn.add_noise1(matrix, chance, white, int(lw), int(rw), black, int(lb), int(rb))
            show_mono2(matrix_noise)
        elif t == 3:
            matrix_noise = nn.add_noise3(matrix, chance, white, int(lw), int(rw))
            show_mono2(matrix_noise)
        elif t == 4:
            matrix_noise = nn.add_noise4(matrix, chance, white, int(lw), int(rw), black, int(lb), int(rb))
            show_mono2(matrix_noise)

    def get_digit():
        global digit, matrix
        digit = enr1.get()
        c = type_digit.get()

        if c == 1:
            matrix = nn.convert_to_matrix(digit, 0, 1)
            text1 = show_binary(matrix)
            mess = tk.Message(window, text=text1, width=1000, font=('Consolas', 11))
            mess.place(x=900, y=5)

        elif c == 2:
            tk.Label(window, text='Белый:', font=('arial', 14)).place(x=510, y=15)
            enr_white = tk.Entry(window, width=5, font=('arial', 14))
            enr_white.place(x=600, y=15)
            tk.Label(window, text='Черный:', font=('arial', 14)).place(x=510, y=50)
            enr_black = tk.Entry(window, width=5, font=('arial', 14))
            enr_black.place(x=600, y=55)
            btn2 = tk.Button(window, text='->', width=3, font=('arial', 14),
                             command=lambda: show_mono(digit, enr_white.get(), enr_black.get()))
            btn2.place(x=680, y=25)

    def get_shift(t):
        global digit
        if t == 1 or t == 3:
            tk.Label(window, text='Вероятность:', font=('arial', 14)).place(x=340, y=145)
            enr2 = tk.Entry(window, width=5, font=('arial', 14))
            enr2.place(x=475, y=147)
            tk.Label(window, text='Белый:', font=('arial', 14)).place(x=20, y=230)
            tk.Label(window, text='Сдвиг влево:', font=('arial', 13)).place(x=120, y=233)
            enr4 = tk.Entry(window, width=5, font=('arial', 14))
            enr4.place(x=240, y=233)
            tk.Label(window, text='Сдвиг вправо:', font=('arial', 13)).place(x=120, y=273)
            enr5 = tk.Entry(window, width=5, font=('arial', 14))
            enr5.place(x=240, y=273)
            if t == 1:
                tk.Label(window, text='Черный:', font=('arial', 14)).place(x=320, y=230)
                tk.Label(window, text='Сдвиг влево:', font=('arial', 13)).place(x=420, y=233)
                enr6 = tk.Entry(window, width=5, font=('arial', 14))
                enr6.place(x=540, y=233)
                tk.Label(window, text='Сдвиг вправо:', font=('arial', 13)).place(x=420, y=273)
                enr7 = tk.Entry(window, width=5, font=('arial', 14))
                enr7.place(x=540, y=273)
                btn3 = tk.Button(window, text='->', width=3, font=('arial', 14),
                                 command=lambda: get_noise_monochrome(digit, t, enr4.get(), enr5.get(),
                                                                      enr6.get(), enr7.get(), float(enr2.get())))
                btn3.place(x=620, y=245)
            elif t == 3:
                btn3 = tk.Button(window, text='->', width=3, font=('arial', 14),
                                 command=lambda: get_noise_monochrome(digit, t, enr4.get(), enr5.get(), 0, 0,
                                                                      float(enr2.get())))
                btn3.place(x=350, y=245)

        elif t == 2:
            tk.Label(window, text='Белый:', font=('arial', 14)).place(x=340, y=110)
            tk.Label(window, text='Черный:', font=('arial', 14)).place(x=340, y=200)
            tk.Label(window, text='Сдвиг влево:', font=('arial', 13)).place(x=430, y=113)
            enr4 = tk.Entry(window, width=5, font=('arial', 14))
            enr4.place(x=550, y=113)
            tk.Label(window, text='Сдвиг вправо:', font=('arial', 13)).place(x=430, y=153)
            enr5 = tk.Entry(window, width=5, font=('arial', 14))
            enr5.place(x=550, y=153)
            tk.Label(window, text='Сдвиг влево:', font=('arial', 13)).place(x=430, y=203)
            enr6 = tk.Entry(window, width=5, font=('arial', 14))
            enr6.place(x=550, y=203)
            tk.Label(window, text='Сдвиг вправо:', font=('arial', 13)).place(x=430, y=243)
            enr7 = tk.Entry(window, width=5, font=('arial', 14))
            enr7.place(x=550, y=243)
            btn0 = tk.Button(window, text='->', width=3, font=('arial', 14),
                             command=lambda: get_noise_monochrome(digit, t, enr4.get(),
                                                                  enr5.get(), enr6.get(), enr7.get()))
            btn0.place(x=635, y=170)
        elif t == 4:
            tk.Label(window, text='Вероятность:', font=('arial', 14)).place(x=340, y=145)
            enr2 = tk.Entry(window, width=5, font=('arial', 14))
            enr2.place(x=475, y=147)
            tk.Label(window, text='Белый:', font=('arial', 14)).place(x=20, y=230)
            tk.Label(window, text='[ ', font=('arial', 15)).place(x=110, y=230)
            enr4 = tk.Entry(window, width=3, font=('arial', 14))
            enr4.place(x=130, y=233)
            tk.Label(window, text=' , ', font=('arial', 17)).place(x=170, y=232)
            enr5 = tk.Entry(window, width=3, font=('arial', 14))
            enr5.place(x=200, y=233)
            tk.Label(window, text=' ]', font=('arial', 15)).place(x=240, y=230)

            tk.Label(window, text='Черный:', font=('arial', 14)).place(x=320, y=230)
            tk.Label(window, text='[ ', font=('arial', 15)).place(x=410, y=230)
            enr6 = tk.Entry(window, width=3, font=('arial', 14))
            enr6.place(x=430, y=233)
            tk.Label(window, text=' , ', font=('arial', 17)).place(x=470, y=232)
            enr7 = tk.Entry(window, width=3, font=('arial', 14))
            enr7.place(x=500, y=233)
            tk.Label(window, text=' ]', font=('arial', 15)).place(x=540, y=230)
            btn3 = tk.Button(window, text='->', width=3, font=('arial', 14),
                             command=lambda: get_noise_monochrome(digit, t, enr4.get(), enr5.get(),
                                                                  enr6.get(), enr7.get(), float(enr2.get())))
            btn3.place(x=620, y=225)

    # Выбираем способ зашумления
    def select_noise():
        c = type_digit.get()
        type_noise = tk.IntVar(value=0)
        if c == 1:
            rbn1 = tk.Radiobutton(window, text="Полное", font=('arial', 14), variable=type_noise, value=1)
            rbn2 = tk.Radiobutton(window, text="Только фон", font=('arial', 14), variable=type_noise, value=2)
            rbn1.place(x=90, y=130)
            rbn2.place(x=90, y=165)
            tk.Label(window, text='Вероятность:', font=('arial', 14)).place(x=250, y=145)
            enr2 = tk.Entry(window, width=5, font=('arial', 14))
            enr2.place(x=385, y=145)
            b = tk.Button(window, text='->', width=3, font=('arial', 14),
                          command=lambda: get_noise_binary(type_noise.get(), enr2.get()))
            b.place(x=460, y=140)
        elif c == 2:
            rbn1 = tk.Radiobutton(window, text="Полное", font=('arial', 13),
                                  variable=type_noise, value=1)
            rbn2 = tk.Radiobutton(window, text='"Равномерное"', font=('arial', 13),
                                  variable=type_noise, value=2)
            rbn3 = tk.Radiobutton(window, text="Только фон", font=('arial', 13),
                                  variable=type_noise, value=3)
            rbn4 = tk.Radiobutton(window, text="Произвольный отрезок", font=('arial', 13),
                                  variable=type_noise, value=4)
            rbn1.place(x=90, y=110)
            rbn2.place(x=90, y=135)
            rbn3.place(x=90, y=160)
            rbn4.place(x=90, y=185)
            bt3 = tk.Button(window, text='->', width=3, font=('arial', 14),
                            command=lambda: get_shift(type_noise.get()))
            bt3.place(x=280, y=140)

    # Метрики
    def show_distance():
        lst = nn.make_list(matrix_noise)
        l1 = nn.square_euclidean_distance(lst, white, black)
        l2 = nn.manhattan_distance(lst, white, black)
        l3 = nn.chebyshev_distance(lst, white, black)
        tk.Message(window, text='Квадрат евклидова\nрасстояния до:',
                   width=1000, font=('Consolas', 13)).place(x=100, y=310)
        text1 = ''
        for i in range(len(l1)):
            text1 += (str(i) + " : " + str(l1[i]) + '\n')

        mess = tk.Message(window, text=text1, width=1000, font=('Consolas', 13))
        mess.place(x=100, y=355)
        tk.Message(window, text='Манхэттенское\nрасстояние до:',
                   width=1000, font=('Consolas', 13)).place(x=315, y=310)
        text2 = ''
        for i in range(len(l2)):
            text2 += (str(i) + " : " + str(l2[i]) + '\n')

        mess = tk.Message(window, text=text2, width=1000, font=('Consolas', 13))
        mess.place(x=315, y=355)

        tk.Label(window, text='Результат: ' + str(l1.index(min(l1))),
                 font=('arial', 13)).place(x=125, y=565)
        tk.Label(window, text='Результат: ' + str(l2.index(min(l2))),
                 font=('arial', 13)).place(x=335, y=565)

        tk.Message(window, text='Расстояние\nЧебышёва до:',
                   width=1000, font=('Consolas', 13)).place(x=515, y=310)
        text3 = ''
        for i in range(len(l3)):
            text3 += (str(i) + " : " + str(l3[i]) + '\n')

        mess = tk.Message(window, text=text3, width=1000, font=('Consolas', 13))
        mess.place(x=515, y=355)
        tk.Label(window, text='Результат: ' + str(l3.index(min(l3))),
                 font=('arial', 13)).place(x=535, y=565)

    window = tk.Tk()
    photo = tk.PhotoImage(file="icon-1.png")
    window.iconphoto(False, photo)

    window.title('Метод ближайшего соседа')
    window.geometry('1250x620+10+10')
    window.resizable(False, False)

    tk.Label(window, text='Цифра:', font=('arial', 14)).place(x=20, y=20)
    enr1 = tk.Entry(window, width=5, font=('arial', 14))
    enr1.place(x=100, y=22)

    type_digit = tk.IntVar(value=0)
    tk.Label(window, text='Тип:', font=('arial', 14)).place(x=180, y=20)
    rb1 = tk.Radiobutton(window, text="Бинарное", font=('arial', 14),
                         variable=type_digit, value=1)
    rb2 = tk.Radiobutton(window, text="В градациях серого", font=('arial', 14),
                         variable=type_digit, value=2)
    rb1.place(x=240, y=15)
    rb2.place(x=240, y=50)

    btn1 = tk.Button(window, text='->', width=3, font=('arial', 14), command=get_digit)
    btn1.place(x=450, y=25)

    # Добавление шума
    btn2 = tk.Button(window, text='Ш', width=3, font=('arial', 14), command=select_noise)
    btn2.place(x=20, y=140)

    # Итоговый вывод
    btn3 = tk.Button(window, text='БС', width=3, font=('arial', 14), command=show_distance)
    btn3.place(x=20, y=400)

    # Перезапуск
    btn4 = tk.Button(window, text='O', width=3, font=('arial', 14), command=quit)
    btn4.place(x=680, y=560)

    # Выход
    btn5 = tk.Button(window, text='X', width=3, font=('arial', 14), command=exit)
    btn5.place(x=740, y=560)

    window.mainloop()


while runing:
    run()
