"""Plot components promoted from QWidget in the Qt Designer forms."""
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStyle
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        super().__init__(Figure())
        self.setParent(parent)


class PlotToolbar(QWidget):
    """Designer owns its position; Matplotlib supplies its standard plot actions."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._toolbar = None

    def bind_canvas(self, canvas):
        if self._toolbar is None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self._toolbar = NavigationToolbar2QT(canvas, self)
            layout.addWidget(self._toolbar)


class AspectRatioLabel(QLabel):
    """A Designer pixmap that fits its container without distorting the image."""

    def sizeHint(self):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return super().sizeHint()
        return pixmap.size().scaled(480, 100, Qt.AspectRatioMode.KeepAspectRatio)

    def minimumSizeHint(self):
        return QSize(0, 0)

    def imageRect(self):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return self.contentsRect()
        size = pixmap.size().scaled(self.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QStyle.alignedRect(self.layoutDirection(), self.alignment(), size, self.contentsRect())

    def paintEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(self.imageRect(), pixmap)
