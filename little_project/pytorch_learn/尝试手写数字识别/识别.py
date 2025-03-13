import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QColorDialog, QLabel
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint

# Define the DigitRecognizer model
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DrawingBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(2000, 2000)
        self.setStyleSheet("background-color: brown;")
        self.pen_color = QColor(Qt.black)
        self.last_point = QPoint()
        self.drawing = False
        self.path = QPainterPath()  # Initialize path

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(self.pen_color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
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

        self.recognize_button = QPushButton("Recognize Digit", self)
        self.recognize_button.clicked.connect(self.recognize_digit)
        self.recognize_button.setFixedSize(150, 30)
        self.recognize_button.move(10, 50)

        self.result_label = QLabel(self)
        self.result_label.setFixedSize(150, 30)
        self.result_label.move(10, 90)

        self.model = DigitRecognizer()
        self.model.load_state_dict(
        self.torch.load('digit_recognizer.pth', map_location=torch.device('cuda:0'), weights_only=True))

        self.model.eval()

    def recognize_digit(self):
        image = self.drawing_board.get_image()
        image = image.scaled(28, 28).convertToFormat(QImage.Format_Grayscale8)
        buffer = image.bits().asstring(image.width() * image.height())
        tensor = torch.frombuffer(buffer, dtype=torch.uint8).float().view(1, 1, 28, 28) / 255.0
        transform = transforms.Normalize((0.5,), (0.5,))
        tensor = transform(tensor)

        with torch.no_grad():
            output = self.model(tensor)
            prediction = torch.argmax(output, dim=1).item()
            self.result_label.setText(f"Digit: {prediction}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())