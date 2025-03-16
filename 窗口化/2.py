import tkinter as tk

from tkinter import *

win = Tk()

win.title("Python Window")
win.geometry("300x400+200+200")

label = Label(win,text="我是wilkins", bg="YellowGreen")
label.pack()

def window():
    win2 = Tk()
    win2.title("dinadadida")
    win2.mainloop()


button = Button(win, text="我是按钮", height=3, width=10, bg="red", command=window)
button.pack()



win.mainloop()