version = "1.2"

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox, QInputDialog
from PyQt5.QtWidgets import QLineEdit, QComboBox, QTabWidget, QPlainTextEdit, QFileDialog
from PyQt5.QtCore import Qt, QCoreApplication, QUrl, QSize, QObject, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QImage, QPalette
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from .Utils import GetScreenSize

class Communicate(QObject):
    signal = pyqtSignal(str)
    def connect(self, foo):
        self.signal.connect(foo)

    def emit(self, message):
        self.signal.emit(message)

W_SCREEN, H_SCREEN = GetScreenSize()

class App(QApplication):
    def __init__(self, title, listparam):
        super().__init__(listparam)
        QCoreApplication.setApplicationName(title)

    def run(self, window, show_mode = "show"):
        "show_mode = 'show' or 'not show'"
        self.currentWindow = window
        if (show_mode == 'show'):
            self.currentWindow.show()
        self.exec_()

    def switch(self, newWindow, show_mode = 'show', hide_mode = 'close'):
        """
        switch to another window and hide current window

        show_mode = 'show' or 'not show'
            if new Window is showed before, choose show_mode is 'not show'
        hide_mode = 'hide' or 'close'
            close will terminate current window, otherwise, hide only make it to be invisible
        """
        if hide_mode == 'close':
            self.currentWindow.close()
        else:
            self.currentWindow.hide()

        if (show_mode == 'show'):
            newWindow.show()
        self.currentWindow = newWindow

class Window(QWidget):
    def __init__(self, width, height, dx = None, dy = None, parent = None):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.resize(width, height)
        if dx == None:
            dx = (W_SCREEN - width) / 2

        if dy == None:
            dy = int( 0.8 * ( H_SCREEN - height) / 2 )
            
        self.move(dx, dy)

    def resize(self, x, y):
        self.width = x
        self.height = y
        super().resize(x, y)

    def getSize(self):
        return self.width, self.height

    def setBackGroundImg(self, imgpath):
        self.background = Button("", self.width, self.height, 0, 0, self)
        self.background.setIcon(QIcon(imgpath))
        self.background.setIconSize(QSize(self.width, self.height))

class Button(QPushButton):
    def __init__(self, text, width, height, dx, dy, parent, border = False):
        self.width = width
        self.height = height
        super().__init__(text, parent)
        self.resize(width, height)
        self.move(dx, dy)
        font = QFont('SogoeUI', 10, 2)
        self.setFont(font)

        if (border):
            self.setStyleSheet("border : 1px solid black")

    def SetClickedEvent(self, method):
        self.clicked.connect(method)

    def resize(self, x, y):
        self.width = x
        self.height = y
        super().resize(x, y)

    def getSize(self):
        return self.width, self.height

class Label(QLabel):
    def __init__(self, text, width, height, dx, dy, parent, border = False):
        self.width = width
        self.height = height
        font = QFont('Sogoe UI', 11, 2)
        super().__init__(text, parent)
        self.resize(width, height)
        self.move(dx, dy)
        self.setFont(font)
        self.setAlignment(Qt.AlignCenter)

        if (border):
            self.setStyleSheet("border : 1px solid black")

    def resize(self, x, y):
        self.width = x
        self.height = y
        super().resize(x, y)

    def getSize(self):
        return self.width, self.height


class ComboBox(QComboBox):
    def __init__(self, items, width, height, dx, dy, parent):
        self.width = width
        self.height = height
        super().__init__(parent)
        font = QFont('Sogoe UI', 11, 2)
        self.setFont(font)
        self.DefautSep = '|'
        itemlist = items.split(self.DefautSep)
        for item in itemlist:
            self.addItem(item)

        self.resize(width, height)
        self.move(dx, dy)

    def SetDefautSep(self, newSep):
        self.DefautSep = newSep

    def addItems(self, items):
        itemlist = items.split(self.DefautSep)
        for item in itemlist:
            self.addItem(item)

    def get(self):
        return self.currentText().strip()

    def resize(self, x, y):
        self.width = x
        self.height = y
        super().resize(x, y)

    def getSize(self):
        return self.width, self.height

class InputBox(QLineEdit):
    def __init__(self, text, width, height, dx, dy, parent):
        self.width = width
        self.height = height
        super().__init__(text, parent)
        self.resize(width, height)
        self.move(dx, dy)

        font = QFont('Sogoe UI', 11, 2)
        self.setFont(font)
        self.setAlignment(Qt.AlignCenter)

    def get(self):
        return self.displayText().strip()

    def resize(self, x, y):
        self.width = x
        self.height = y
        super().resize(x, y)

    def getSize(self):
        return self.width, self.height

class TabWidget(QTabWidget):
    def __init__(self, width, height, dx, dy, parent):
        self.width = width
        self.height = height
        super().__init__(parent)
        self.resize(width, height)
        self.move(dx, dy)

    
    def resize(self, x, y):
        self.width = x
        self.height = y
        super().resize(x, y)

    def getSize(self):
        return self.width, self.height

class TextEdit(QPlainTextEdit):
    def __init__(self, text, width, height, dx, dy, parent):
        self.width = width
        self.height = height
        super().__init__(text, parent)
        self.resize(width, height)
        self.move(dx, dy)
        font = QFont('Sogoe UI', 11, 2)
        self.setFont(font)
        #self.setAlignment(Qt.AlignCenter)

    def resize(self, width, height):
        super().resize(width, height)
        self.width = width
        self.height = height
    
    def getSize(self):
        return self.width, self.height

    def addLine(self, line):
        current_message = self.toPlainText()
        self.setPlainText(current_message + "\n" + line)

class MessageBox(QMessageBox):
    def __init__(self, text, parent):
        super().__init__(parent)
        self.setText(text)
        self.show()
        self.exec_()

class Table(QTableWidget):
    def __init__(self, width, height, dx, dy, nrows, ncolumns, parent):
        super().__init__(parent)
        self.setColumnCount(ncolumns)
        self.setRowCount(nrows)
        self.move(dx, dy)
        self.resize(width, height)
    
    def setItem_(self, x, y, item):
        super().setItem(x, y, QTableWidgetItem(item))

