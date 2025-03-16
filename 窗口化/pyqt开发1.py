import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt6.QtGui import QIcon

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