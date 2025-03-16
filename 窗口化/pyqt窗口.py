import sys
from PyQt6.QtWidgets import QApplication,QWidget,QVBoxLayout,QLineEdit,QPushButton,QLabel
from PyQt6 import QtCore
from PyQt6.QtGui import QIcon

class LoginWindow(QWidget):
    loginSuccessSignal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.username_input = QLineEdit(self)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)

        login_button = QPushButton('Login', self)
        login_button.clicked.connect(self.login)

        layout.addWidget(QLabel('Username:'))
        layout.addWidget(self.username_input)
        layout.addWidget(QLabel('Password:'))
        layout.addWidget(self.password_input)
        layout.addWidget(login_button)

        self.setGeometry(300, 300, 400, 200)
        self.setWindowTitle('Login Window')

    def Login(self):
        if self.username_input.text() == '1' and self.password_input.text() == '1':
            self.loginSuccessSignal.emit()
            self.close()

class YourMainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.login_window = LoginWindow()
        self.login_window.loginSuccessSignal.connect(self.showMainWindow)

    def showMainWindow(self):
        self.setGeometry(300, 300, 400, 200)
        self.setWindowTitle('你好')
        self.setWindowIcon(QIcon('logo.png'))
        self.show()
if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = YourMainWindow
    main_window.showMainWindow()

    sys.exit(app.exec())

