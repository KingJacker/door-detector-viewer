from PySide6.QtWidgets import QWidget, QHBoxLayout, QSlider, QLabel
from PySide6.QtCore import Signal, Qt


class RangeSlider(QWidget):
    rangeChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, 100)
        self.min_slider.setValue(0)
        self.min_slider.valueChanged.connect(self._on_min_changed)

        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(0, 100)
        self.max_slider.setValue(100)
        self.max_slider.valueChanged.connect(self._on_max_changed)

        self.min_label = QLabel("0")
        self.min_label.setMinimumWidth(30)

        self.max_label = QLabel("100")
        self.max_label.setMinimumWidth(30)

        layout.addWidget(self.min_label)
        layout.addWidget(self.min_slider)
        layout.addWidget(QLabel("-"))
        layout.addWidget(self.max_slider)
        layout.addWidget(self.max_label)

    def setRange(self, min_val, max_val):
        self.min_slider.setRange(min_val, max_val)
        self.max_slider.setRange(min_val, max_val)

    def setValue(self, min_val, max_val):
        self.min_slider.setValue(min_val)
        self.max_slider.setValue(max_val)
        self.min_label.setText(str(min_val))
        self.max_label.setText(str(max_val))

    def value(self):
        return self.min_slider.value(), self.max_slider.value()

    def _on_min_changed(self, value):
        if value > self.max_slider.value():
            self.max_slider.setValue(value)
        self.min_label.setText(str(value))
        self.rangeChanged.emit(self.min_slider.value(), self.max_slider.value())

    def _on_max_changed(self, value):
        if value < self.min_slider.value():
            self.min_slider.setValue(value)
        self.max_label.setText(str(value))
        self.rangeChanged.emit(self.min_slider.value(), self.max_slider.value())
