import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QColorDialog, QLabel, QMessageBox
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath, QImage
from PyQt5.QtCore import Qt, QPoint

# Define the DigitRecognizer model with three hidden layers

import torch.nn as nn

class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 64)  # Ensure this matches the saved model weights
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)  # Ensure this matches the saved model weights
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DrawingBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(600, 600)
        self.setStyleSheet("background-color: brown;")
        self.pen_color = QColor(Qt.black)
        self.pen_width = 20  # Set pen width to 20
        self.last_point = QPoint()
        self.drawing = False
        self.path = QPainterPath()  # Initialize path

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(self.path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            self.path.moveTo(self.last_point)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            self.path.lineTo(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.parent().recognize_digit()  # Automatically recognize digit after drawing

    def set_pen_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.pen_color = color

    def get_image(self):
        image = QImage(self.size(), QImage.Format_RGB32)
        painter = QPainter(image)
        self.render(painter)
        return image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handwriting Board")
        self.drawing_board = DrawingBoard()
        self.setCentralWidget(self.drawing_board)

        self.color_button = QPushButton("Change Pen Color", self)
        self.color_button.clicked.connect(self.drawing_board.set_pen_color)
        self.color_button.setFixedSize(150, 30)
        self.color_button.move(10, 10)

        self.result_label = QLabel(self)
        self.result_label.setFixedSize(150, 30)
        self.result_label.move(10, 50)

        self.model = DigitRecognizer()
        self.model.load_state_dict(torch.load('digit_recognizer.pth', map_location=torch.device('cuda:0'), weights_only=True))
        self.model.eval()

    def recognize_digit(self):
        image = self.drawing_board.get_image()
        image = image.scaled(28, 28).convertToFormat(QImage.Format_Grayscale8)
        buffer = image.bits().asstring(image.width() * image.height())
        buffer_copy = np.frombuffer(buffer, dtype=np.uint8).copy()  # Copy the buffer to make it writable
        tensor = torch.from_numpy(buffer_copy).float().view(1, 1, 28, 28) / 255.0
        transform = transforms.Normalize((0.5,), (0.5,))
        tensor = transform(tensor)

        with torch.no_grad():
            output = self.model(tensor)
            prediction = torch.argmax(output, dim=1).item()

        # Display the result in a popup window
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Recognized Digit: {prediction}")
        msg.setWindowTitle("Recognition Result")
        msg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

# import sys
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QColorDialog, QLabel, QMessageBox
# from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath, QImage, QPixmap
# from PyQt5.QtCore import Qt, QPoint
#
# def recognize_digit(self):
#     image = self.drawing_board.get_image()
#     image = image.scaled(28, 28).convertToFormat(QImage.Format_Grayscale8)
#     buffer = image.bits().asstring(image.width() * image.height())
#     buffer_copy = np.frombuffer(buffer, dtype=np.uint8).copy()  # Copy the buffer to make it writable
#     tensor = torch.from_numpy(buffer_copy).float().view(1, 1, 28, 28) / 255.0
#     transform = transforms.Normalize((0.5,), (0.5,))
#     tensor = transform(tensor)
#
#     with torch.no_grad():
#         output = self.model(tensor)
#         prediction = torch.argmax(output, dim=1).item()
#
#     # Display the result in a popup window
#     msg = QMessageBox()
#     msg.setIcon(QMessageBox.Information)
#     msg.setText(f"Recognized Digit: {prediction}")
#     msg.setWindowTitle("Recognition Result")
#     msg.exec_()
#
# # Define the DigitRecognizer model
# class DigitRecognizer(nn.Module):
#     def __init__(self):
#         super(DigitRecognizer, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64*7*7, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64*7*7)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
# class DrawingBoard(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setFixedSize(2000, 2000)
#         self.setStyleSheet("background-color: brown;")
#         self.pen_color = QColor(Qt.black)
#         self.pen_width = 20  # Set pen width to 10
#         self.last_point = QPoint()
#         self.drawing = False
#         self.path = QPainterPath()  # Initialize path
#
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setPen(QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#         painter.drawPath(self.path)
#
#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.drawing = True
#             self.last_point = event.pos()
#             self.path.moveTo(self.last_point)
#
#     def mouseMoveEvent(self, event):
#         if event.buttons() & Qt.LeftButton and self.drawing:
#             self.path.lineTo(event.pos())
#             self.update()
#
#     def mouseReleaseEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.drawing = False
#             self.parent().recognize_digit()  # Automatically recognize digit after drawing
#
#     def set_pen_color(self):
#         color = QColorDialog.getColor()
#         if color.isValid():
#             self.pen_color = color
#
#     def get_image(self):
#         image = QImage(self.size(), QImage.Format_RGB32)
#         painter = QPainter(image)
#         self.render(painter)
#         return image
#
#
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Handwriting Board")
#         self.drawing_board = DrawingBoard()
#         self.setCentralWidget(self.drawing_board)
#
#         self.color_button = QPushButton("Change Pen Color", self)
#         self.color_button.clicked.connect(self.drawing_board.set_pen_color)
#         self.color_button.setFixedSize(150, 30)
#         self.color_button.move(10, 10)
#
#         self.result_label = QLabel(self)
#         self.result_label.setFixedSize(150, 30)
#         self.result_label.move(10, 50)
#
#         self.model = DigitRecognizer()
#         self.model.load_state_dict(torch.load('digit_recognizer.pth', map_location=torch.device('cpu'), weights_only=True))
#         self.model.eval()
#
#     def recognize_digit(self):
#         image = self.drawing_board.get_image()
#         image = image.scaled(28, 28).convertToFormat(QImage.Format_Grayscale8)
#         buffer = image.bits().asstring(image.width() * image.height())
#         buffer_copy = np.frombuffer(buffer, dtype=np.uint8).copy()  # Copy the buffer to make it writable
#         tensor = torch.from_numpy(buffer_copy).float().view(1, 1, 28, 28) / 255.0
#         transform = transforms.Normalize((0.5,), (0.5,))
#         tensor = transform(tensor)
#
#         with torch.no_grad():
#             output = self.model(tensor)
#             prediction = torch.argmax(output, dim=1).item()
#
#         # Display the result in a popup window
#         msg = QMessageBox()
#         msg.setIcon(QMessageBox.Information)
#         msg.setText(f"Recognized Digit: {prediction}")
#         msg.setWindowTitle("Recognition Result")
#         msg.exec_()
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec_())



# import sys# class DrawingBoard(QWidget):
# #     def __init__(self):
# #         super().__init__()
# #         self.setFixedSize(2000, 2000)
# #         self.setStyleSheet("background-color: brown;")
# #         self.pen_color = QColor(Qt.black)
# #         self.last_point = QPoint()
# #         self.drawing = False
# #         self.path = QPainterPath()  # Initialize path
# #
# #     def paintEvent(self, event):
# #         painter = QPainter(self)
# #         painter.setPen(QPen(self.pen_color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
# #         painter.drawPath(self.path)
# #
# #     def mousePressEvent(self, event):
# #         if event.button() == Qt.LeftButton:
# #             self.drawing = True
# #             self.last_point = event.pos()
# #             self.path.moveTo(self.last_point)
# #
# #     def mouseMoveEvent(self, event):
# #         if event.buttons() & Qt.LeftButton and self.drawing:
# #             self.path.lineTo(event.pos())
# #             self.update()
# #
# #     def mouseReleaseEvent(self, event):
# #         if event.button() == Qt.LeftButton:
# #             self.drawing = False
# #             self.parent().recognize_digit()  # Automatically recognize digit after drawing
# #
# #     def set_pen_color(self):
# #         color = QColorDialog.getColor()
# #         if color.isValid():
# #             self.pen_color = color
# #
# #     def get_image(self):
# #         image = QImage(self.size(), QImage.Format_RGB32)
# #         painter = QPainter(image)
# #         self.render(painter)
# #         return image
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QColorDialog, QLabel, QMessageBox
# from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath, QImage, QPixmap
# from PyQt5.QtCore import Qt, QPoint
#
# def recognize_digit(self):
#     image = self.drawing_board.get_image()
#     image = image.scaled(28, 28).convertToFormat(QImage.Format_Grayscale8)
#     buffer = image.bits().asstring(image.width() * image.height())
#     buffer_copy = np.frombuffer(buffer, dtype=np.uint8).copy()  # Copy the buffer to make it writable
#     tensor = torch.from_numpy(buffer_copy).float().view(1, 1, 28, 28) / 255.0
#     transform = transforms.Normalize((0.5,), (0.5,))
#     tensor = transform(tensor)
#
#     with torch.no_grad():
#         output = self.model(tensor)
#         prediction = torch.argmax(output, dim=1).item()
#
#     # Display the result in a popup window
#     msg = QMessageBox()
#     msg.setIcon(QMessageBox.Information)
#     msg.setText(f"Recognized Digit: {prediction}")
#     msg.setWindowTitle("Recognition Result")
#     msg.exec_()
#
# # Define the DigitRecognizer model
# class DigitRecognizer(nn.Module):
#     def __init__(self):
#         super(DigitRecognizer, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64*7*7, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()
#
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64*7*7)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# class DrawingBoard(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setFixedSize(2000, 2000)
#         self.setStyleSheet("background-color: brown;")
#         self.pen_color = QColor(Qt.black)
#         self.last_point = QPoint()
#         self.drawing = False
#         self.path = QPainterPath()  # Initialize path
#
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setPen(QPen(self.pen_color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#         painter.drawPath(self.path)
#
#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.drawing = True
#             self.last_point = event.pos()
#             self.path.moveTo(self.last_point)
#
#     def mouseMoveEvent(self, event):
#         if event.buttons() & Qt.LeftButton and self.drawing:
#             self.path.lineTo(event.pos())
#             self.update()
#
#     def mouseReleaseEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.drawing = False
#             self.parent().recognize_digit()  # Automatically recognize digit after drawing
#
#     def set_pen_color(self):
#         color = QColorDialog.getColor()
#         if color.isValid():
#             self.pen_color = color
#
#     def get_image(self):
#         image = QImage(self.size(), QImage.Format_RGB32)
#         painter = QPainter(image)
#         self.render(painter)
#         return image
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Handwriting Board")
#         self.drawing_board = DrawingBoard()
#         self.setCentralWidget(self.drawing_board)
#
#         self.color_button = QPushButton("Change Pen Color", self)
#         self.color_button.clicked.connect(self.drawing_board.set_pen_color)
#         self.color_button.setFixedSize(150, 30)
#         self.color_button.move(10, 10)
#
#         self.result_label = QLabel(self)
#         self.result_label.setFixedSize(150, 30)
#         self.result_label.move(10, 50)
#
#         self.model = DigitRecognizer()
#         self.model.load_state_dict(torch.load('digit_recognizer.pth', map_location=torch.device('cpu'), weights_only=True))
#         self.model.eval()
#
#     def recognize_digit(self):
#         image = self.drawing_board.get_image()
#         image = image.scaled(28, 28).convertToFormat(QImage.Format_Grayscale8)
#         buffer = image.bits().asstring(image.width() * image.height())
#         tensor = torch.frombuffer(buffer, dtype=torch.uint8).float().view(1, 1, 28, 28) / 255.0
#         transform = transforms.Normalize((0.5,), (0.5,))
#         tensor = transform(tensor)
#
#         with torch.no_grad():
#             output = self.model(tensor)
#             prediction = torch.argmax(output, dim=1).item()
#
#         # Display the result in a popup window
#         msg = QMessageBox()
#         msg.setIcon(QMessageBox.Information)
#         msg.setText(f"Recognized Digit: {prediction}")
#         msg.setWindowTitle("Recognition Result")
#         msg.exec_()
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec_())

# import sys
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QColorDialog, QLabel
# from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath, QImage, QPixmap
# from PyQt5.QtCore import Qt, QPoint
#
# # Define the DigitRecognizer model
# class DigitRecognizer(nn.Module):
#     def __init__(self):
#         super(DigitRecognizer, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64*7*7, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64*7*7)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# class DrawingBoard(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setFixedSize(2000, 2000)
#         self.setStyleSheet("background-color: brown;")
#         self.pen_color = QColor(Qt.black)
#         self.last_point = QPoint()
#         self.drawing = False
#         self.path = QPainterPath()  # Initialize path
#
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setPen(QPen(self.pen_color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#         painter.drawPath(self.path)
#
#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.drawing = True
#             self.last_point = event.pos()
#             self.path.moveTo(self.last_point)
#
#     def mouseMoveEvent(self, event):
#         if event.buttons() & Qt.LeftButton and self.drawing:
#             self.path.lineTo(event.pos())
#             self.update()
#
#     def mouseReleaseEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.drawing = False
#
#     def set_pen_color(self):
#         color = QColorDialog.getColor()
#         if color.isValid():
#             self.pen_color = color
#
#     def get_image(self):
#         image = QImage(self.size(), QImage.Format_RGB32)
#         painter = QPainter(image)
#         self.render(painter)
#         return image
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Handwriting Board")
#         self.drawing_board = DrawingBoard()
#         self.setCentralWidget(self.drawing_board)
#
#         self.color_button = QPushButton("Change Pen Color", self)
#         self.color_button.clicked.connect(self.drawing_board.set_pen_color)
#         self.color_button.setFixedSize(150, 30)
#         self.color_button.move(10, 10)
#
#         self.recognize_button = QPushButton("Recognize Digit", self)
#         self.recognize_button.clicked.connect(self.recognize_digit)
#         self.recognize_button.setFixedSize(150, 30)
#         self.recognize_button.move(10, 50)
#
#         self.result_label = QLabel(self)
#         self.result_label.setFixedSize(150, 30)
#         self.result_label.move(10, 90)
#
#         self.model = DigitRecognizer()
#         self.model.load_state_dict(torch.load('digit_recognizer.pth', map_location=torch.device('cpu'), weights_only=True))
#         self.model.eval()
#
#     def recognize_digit(self):
#         image = self.drawing_board.get_image()
#         image = image.scaled(28, 28).convertToFormat(QImage.Format_Grayscale8)
#         buffer = image.bits().asstring(image.width() * image.height())
#         tensor = torch.frombuffer(buffer, dtype=torch.uint8).float().view(1, 1, 28, 28) / 255.0
#         transform = transforms.Normalize((0.5,), (0.5,))
#         tensor = transform(tensor)
#
#         with torch.no_grad():
#             output = self.model(tensor)
#             prediction = torch.argmax(output, dim=1).item()
#             self.result_label.setText(f"Digit: {prediction}")
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec_())