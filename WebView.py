version = "1.0"
from PyQt5.QtWebEngineWidgets import QWebEngineView

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