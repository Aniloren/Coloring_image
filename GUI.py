from tkinter import *
from tkinter import filedialog

from PIL import Image, ImageTk

class GUI:
    def __init__(self, image, target):
        self.root = Tk()
        self.root.geometry("1060x650+300+150")
        # создаем рабочую область
        self.frame = Frame(self.root)
        self.frame.grid()

        # Добавим метку
        self.label = Label(self.frame, text="Hello, World!").grid(row=0, column=0)

        self.image = image
        self.image = self.image.resize((256, 256), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image)

        self.target = target
        self.target = self.target.resize((256, 256), Image.ANTIALIAS)
        self.target = ImageTk.PhotoImage(self.target)

        # вставляем кнопку
        self.but = Button(self.frame, text="Кнопка", command=self.my_event_handler).grid(row=0,column=3)

        # Добавим изображение
        self.canvas1 = Canvas(self.root, height=256, width=256)
        self.canvas2 = Canvas(self.root, height=256, width=256)

        self.a_image = self.canvas1.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas1.grid(row=2, column=1)

        self.b_image = self.canvas2.create_image(0, 0, anchor='nw', image=self.target)
        self.canvas2.grid(row=2, column=2)

        self.root.mainloop()

    def my_event_handler(self):
        print("my_event_handler")
        # self.image = target
        # self.photo = ImageTk.PhotoImage(self.image)
        # self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        # self.canvas.grid(row=2, column=1)

