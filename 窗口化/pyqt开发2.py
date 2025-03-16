import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QMainWindow
from PyQt6.QtGui import QScreen, QGuiApplication
from PyQt6.QtGui import QIcon

class MyApplication(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(0, 0, 200, 200)
        self.setWindowTitle('123')

def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setGeometry(100, 100, 300, 200)
    window.setWindowTitle("123")
    window.setWindowIcon(QIcon('logo.png'))

    QLabel("Hello ", window)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()