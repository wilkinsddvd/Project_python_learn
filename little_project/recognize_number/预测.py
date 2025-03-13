import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw
import tkinter as tk
from tkinter import messagebox

# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 加载预训练模型
model = CNN()
state_dict = torch.load('mnist_cnn_model.pth', map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)
model.eval()  # 设置模型为评估模式

# 定义数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 确保图像为单通道（灰度图）
    transforms.Resize((28, 28)),  # 调整图像大小为 28x28
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

# Tkinter 应用程序
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Handwritten Digit Recognition")
        self.canvas = tk.Canvas(self, width=200, height=200, bg='white')
        self.canvas.pack()
        self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit)
        self.button_predict.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
        self.draw.line([x1, y1, x2, y2], fill=0, width=10)

    def predict_digit(self):
        # 捕获当前画布内容
        self.image = ImageOps.invert(self.image)
        image = self.image.resize((28, 28))
        image = transform(image).unsqueeze(0)

        # 进行预测
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)

        # 显示预测结果
        messagebox.showinfo("Prediction", f'The predicted digit is: {predicted.item()}')

        # 重置画布
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.delete("all")

# 运行应用程序
if __name__ == "__main__":
    app = App()
    app.mainloop()

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image, ImageOps, ImageDraw
# import tkinter as tk
# from tkinter import messagebox
#
# # 定义卷积神经网络模型
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.dropout = nn.Dropout(p=0.5)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 7 * 7)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
#
# # 加载预训练模型
# model = CNN()
# state_dict = torch.load('mnist_cnn_model.pth', map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)
# model.eval()  # 设置模型为评估模式
#
# # 定义数据预处理
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),  # 确保图像为单通道（灰度图）
#     transforms.Resize((28, 28)),  # 调整图像大小为 28x28
#     transforms.ToTensor(),  # 转换为张量
#     transforms.Normalize((0.5,), (0.5,))  # 标准化
# ])
#
# # Tkinter 应用程序
# class App(tk.Tk):
#     def __init__(self):
#         super().__init__()
#
#         self.title("Handwritten Digit Recognition")
#         self.canvas = tk.Canvas(self, width=200, height=200, bg='white')
#         self.canvas.pack()
#         self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit)
#         self.button_predict.pack()
#
#         self.canvas.bind("<B1-Motion>", self.paint)
#
#         self.image = Image.new("L", (200, 200), 255)
#         self.draw = ImageDraw.Draw(self.image)
#
#     def paint(self, event):
#         x1, y1 = (event.x - 1), (event.y - 1)
#         x2, y2 = (event.x + 1), (event.y + 1)
#         self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
#         self.draw.line([x1, y1, x2, y2], fill=0, width=10)
#
#     def predict_digit(self):
#         # 捕获当前画布内容
#         self.image = ImageOps.invert(self.image)
#         image = self.image.resize((28, 28))
#         image = transform(image).unsqueeze(0)
#
#         # 进行预测
#         with torch.no_grad():
#             outputs = model(image)
#             _, predicted = torch.max(outputs.data, 1)
#
#         # 显示预测结果
#         messagebox.showinfo("Prediction", f'The predicted digit is: {predicted.item()}')
#
#         # 重置画布
#         self.image = Image.new("L", (200, 200), 255)
#         self.draw = ImageDraw.Draw(self.image)
#         self.canvas.delete("all")
#
# # 运行应用程序
# if __name__ == "__main__":
#     app = App()
#     app.mainloop()