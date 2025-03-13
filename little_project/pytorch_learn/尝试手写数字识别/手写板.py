import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QColorDialog
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPoint

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QColorDialog
# from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath
# from PyQt5.QtCore import Qt, QPoint
#
# class DrawingBoard(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setFixedSize(800, 600)
#         self.setStyleSheet("background-color: brown;")
#         self.pen_color = QColor(Qt.black)
#         self.last_point = QPoint()
#         self.drawing = False
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
#             self.path = QPainterPath(self.last_point)
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
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec_())