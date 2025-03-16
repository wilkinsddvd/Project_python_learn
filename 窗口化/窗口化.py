import tkinter as tk
from tkinter import messagebox

def login():
    name = username.get()
    pwd = password.get()
    print(name, pwd)
    if name == 'admin' and pwd == '123456':
        print('登录成功')
    else:
        messagebox.showwarning(title='警告', message='登录失败，请检查账户密码是否正确')

root = tk.Tk()

root.geometry('300x180')
root.title('登录页')

username = tk.StringVar()
password = tk.StringVar()

page = tk.Frame(root)
page.pack()

tk.Label(page).grid(row=0, column=0)

tk.Label(page, text='账户:').grid(row=1, column=1)
tk.Entry(page, textvariable=username).grid(row=1, column=2 )

tk.Label(page, text='密码:').grid(row=2, column=1)
tk.Entry(page, textvariable=password).grid(row=2, column=2 )




tk.Button(page, text='登录', command=login).grid(row=3, column=1, pady=10)
tk.Button(page, text='退出', command=page.quit).grid(row=3, column=2)

root.mainloop()
