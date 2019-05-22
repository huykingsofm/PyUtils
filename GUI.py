from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox, QInputDialog
from PyQt5.QtWidgets import QLineEdit, QComboBox, QTabWidget, QPlainTextEdit, QFileDialog
from PyQt5.QtCore import Qt, QCoreApplication, QUrl
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtWebEngineWidgets import QWebEngineView
import wx

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
    def __init__(self, width, height, dx, dy, parent):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.resize(width, height)
        self.move(dx, dy)

    def showOnly(self, widget):
        self.show()
        children = self.findChildren(QWidget)
        if (widget not in children):
            raise Exception("subWidget not in window")

        for child in children:
            child.hide()

        widget.show()

    def switch(self, widget, show_mode = 'show', hide_mode = 'close'):
        """ show another child widget and hide current child widget

            show_mode = 'show' or 'not show'
                if new widget is showed before, choose show_mode is 'not show'
            hide_mode = 'hide' or 'close'
                close will terminate current widget, otherwise, hide only make it to be invisible
        """

        children = self.findChildren(QWidget)
        if (widget not in children):
            raise Exception("Widget not in current window")

        for child in children:
            child.hide() if hide_mode == 'hide' else child.show()

        if (show_mode == 'show'):
            widget.show()

    def resize(self, x, y):
        self.width = x
        self.height = y
        super().resize(x, y)

    def getSize(self):
        return self.width, self.height

class Button(QPushButton):
    def __init__(self, text, width, height, dx, dy, parent):
        self.width = width
        self.height = height
        super().__init__(text, parent)
        self.resize(width, height)
        self.move(dx, dy)
        font = QFont('SogoeUI', 10, 2)
        self.setFont(font)

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

class WebView(QWebEngineView):
    def __init__(self, width, height, dx, dy, parent):
        self.width = width
        self.height = height
        super().__init__(parent)
        self.resize(width, height)
        self.move(dx, dy)

    def resize(self, width, height):
        super().resize(width, height)
        self.width = width
        self.height = height
    
    def getSize(self):
        return self.width, self.height

def GetScreenSize():
    a = wx.App(False)
    return wx.GetDisplaySize()