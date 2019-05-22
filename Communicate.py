from PyQt5.QtCore import QObject, pyqtSignal

class Communicate(QObject):
    signal = pyqtSignal(str)
    def connect(self, foo):
        self.signal.connect(foo)

    def emit(self, message):
        self.signal.emit(message)