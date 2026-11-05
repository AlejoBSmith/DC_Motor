"""Compact coefficient display with an independent, full-precision value."""
import math

from PyQt6 import QtCore, QtWidgets


class CoefficientEdit(QtWidgets.QLineEdit):
    """Show a configured number of decimals, retaining the coefficient text.

    ``rawText`` is used by controller calculations and configuration storage.
    Focusing the field exposes that original text for editing; leaving it restores
    the compact display. An exactly zero coefficient leaves its placeholder visible.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_text = ""
        self._decimals = 2
        self.textEdited.connect(self._capture_edit)

    @QtCore.pyqtProperty(int)
    def decimals(self):
        return self._decimals

    @decimals.setter
    def decimals(self, value):
        self.setDecimals(value)

    def setDecimals(self, value):
        """Setter used by PyQt uic for the Designer property."""
        self._decimals = max(0, min(int(value), 15))
        if not self.hasFocus():
            with QtCore.QSignalBlocker(self):
                super().setText(self._compact_text())

    def rawText(self):
        return self._source_text

    def _capture_edit(self, text):
        self._source_text = text
        self.setToolTip(text)

    def _compact_text(self):
        text = self._source_text.strip()
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            return text
        if not math.isfinite(value):
            return text
        if value == 0:
            return ""
        result = f"{value:.{self._decimals}f}"
        return result[1:] if result.startswith('-') and float(result) == 0 else result

    def setText(self, text):
        text = str(text)
        before_raw, before_display = self._source_text, self.text()
        self._source_text = text
        self.setToolTip(text)
        super().setText(text if self.hasFocus() else self._compact_text())
        # Distinct exact values may have the same rounded representation. The
        # controller and saved configuration must still observe the change.
        if before_raw != text and before_display == self.text():
            self.textChanged.emit(self.text())

    def clear(self):
        self.setText("")

    def focusInEvent(self, event):
        with QtCore.QSignalBlocker(self):
            super().setText(self._source_text)
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        with QtCore.QSignalBlocker(self):
            super().setText(self._compact_text())
