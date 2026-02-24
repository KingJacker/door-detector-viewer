import json
import sys
from pathlib import Path

import numpy as np
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QComboBox,
    QGroupBox,
    QSplitter,
    QSpinBox,
    QCheckBox,
    QFormLayout,
)
from PySide6.QtCore import Qt, QEvent, QTimer
from PySide6.QtGui import QImage, QPixmap

from vo import VisualOdometry
from range_slider import RangeSlider


CONFIG_PATH = Path(__file__).parent / "config.json"


class DataViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Door Detector Data Viewer")
        self.resize(1600, 900)

        self.session_dir = None
        self.frames = []
        self.imu_data = None
        self.current_frame = 0
        self.current_depth = None
        self.current_conf = None
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.next_frame)
        self.playing = False

        self.config = self._load_config()

        self.vo = VisualOdometry(self.config)
        self.vo_results = []

        self.graph_data = {
            "tracked": [],
            "rejected": [],
            "ratio": [],
            "pos_x": [],
            "pos_y": [],
            "pos_z": [],
            "mean_depth": [],
            "mean_conf": [],
            "fps": [],
        }

        self.graph_colors = {
            "tracked": "#00ff00",
            "rejected": "#ff0000",
            "ratio": "#ffff00",
            "pos_x": "#ff00ff",
            "pos_y": "#00ffff",
            "pos_z": "#ff8800",
            "mean_depth": "#88ff88",
            "mean_conf": "#8888ff",
            "fps": "#ffffff",
        }

        self._setup_ui()

    def _load_config(self):
        if CONFIG_PATH.exists():
            try:
                return json.loads(CONFIG_PATH.read_text())
            except Exception:
                pass
        return {
            "conf_threshold": 100,
            "view_mode": "depth",
            "playback_speed": 10,
            "bilateral_enabled": False,
            "bilateral_d": 9,
            "bilateral_sigma_color": 200,
            "bilateral_sigma_space": 20,
            "median_enabled": False,
            "median_k": 5,
            "focal_length_mm": 3.4,
            "sensor_size": "1/6",
            "detector_threshold": 20,
            "rejection_threshold": 8.0,
            "stationary_threshold": 0.5,
            "clahe_enabled": False,
            "clahe_clip_limit": 2.0,
            "clahe_grid_size": 8,
        }

    def _save_config(self):
        CONFIG_PATH.write_text(json.dumps(self.config, indent=2))

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        main_layout.addWidget(splitter)

        left_sidebar = self._create_left_sidebar()
        splitter.addWidget(left_sidebar)

        center_widget = self._create_center_panel()
        splitter.addWidget(center_widget)

        right_sidebar = self._create_right_sidebar()
        splitter.addWidget(right_sidebar)

        splitter.setSizes([250, 900, 300])

        self._populate_sessions()

    def _create_left_sidebar(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        stats_group = QGroupBox("VO Stats")
        stats_layout = QFormLayout(stats_group)

        self.vo_enabled_cb = QCheckBox("Enable Tracking")
        self.vo_enabled_cb.setChecked(True)
        self.vo_enabled_cb.toggled.connect(self._on_vo_enabled_changed)
        stats_layout.addRow("", self.vo_enabled_cb)

        self.tracked_label = QLabel("0")
        stats_layout.addRow("Tracked:", self.tracked_label)
        self.rejected_label = QLabel("0")
        stats_layout.addRow("Rejected:", self.rejected_label)
        self.ratio_label = QLabel("0%")
        stats_layout.addRow("Ratio:", self.ratio_label)
        self.vo_fps_label = QLabel("0")
        stats_layout.addRow("VO FPS:", self.vo_fps_label)

        self.show_tracked_cb = QCheckBox("Show tracked points")
        self.show_tracked_cb.setChecked(False)
        self.show_tracked_cb.toggled.connect(self._on_overlay_toggled)
        stats_layout.addRow("", self.show_tracked_cb)

        self.show_rejected_cb = QCheckBox("Show rejected points")
        self.show_rejected_cb.setChecked(False)
        self.show_rejected_cb.toggled.connect(self._on_overlay_toggled)
        stats_layout.addRow("", self.show_rejected_cb)

        layout.addWidget(stats_group)

        series_group = QGroupBox("Graph Series")
        series_layout = QVBoxLayout(series_group)

        self.series_cbs = {}
        series_options = [
            ("tracked", "Tracked Points"),
            ("rejected", "Rejected Points"),
            ("ratio", "Ratio %"),
            ("pos_x", "Position X"),
            ("pos_y", "Position Y"),
            ("pos_z", "Position Z"),
            ("mean_depth", "Mean Depth"),
            ("mean_conf", "Mean Conf"),
            ("fps", "FPS"),
        ]
        for key, label in series_options:
            color = self.graph_colors.get(key, "#fff")
            cb = QCheckBox(f"■ {label}")
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; }}")
            cb.setChecked(key in ["tracked", "rejected", "ratio"])
            cb.toggled.connect(self._on_series_toggled)
            self.series_cbs[key] = cb
            series_layout.addWidget(cb)

        layout.addWidget(series_group)
        layout.addStretch()

        return widget

    def _create_center_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(480, 360)
        self.image_label.installEventFilter(self)
        layout.addWidget(self.image_label, 3)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        layout.addWidget(self.slider)

        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground("#1e1e1e")
        self.graph_widget.setLabel("bottom", "Frame")
        self.graph_widget.showGrid(x=True, y=True, alpha=0.3)
        self.graph_widget.setMouseTracking(True)
        self.graph_widget.setMouseEnabled(x=False, y=False)

        self.plot_curves = {}
        for key, color in self.graph_colors.items():
            curve = self.graph_widget.plot(pen=color, name=key)
            self.plot_curves[key] = curve
            curve.setVisible(key in ["tracked", "rejected", "ratio"])

        self.graph_tooltip = pg.TextItem(
            anchor=(0, 1), color="w", fill=(50, 50, 50, 200)
        )
        self.graph_widget.addItem(self.graph_tooltip)
        self.graph_tooltip.hide()

        self.graph_widget.scene().sigMouseMoved.connect(self._on_graph_mouse_moved)

        self.playhead_line = pg.InfiniteLine(
            pos=0, pen=pg.mkPen("#ffffff", width=2), movable=False
        )
        self.graph_widget.addItem(self.playhead_line)
        layout.addWidget(self.graph_widget, 2)

        self.range_slider = RangeSlider()
        self.range_slider.setRange(0, 100)
        self.range_slider.setValue(0, 100)
        self.range_slider.rangeChanged.connect(self._on_range_changed)
        layout.addWidget(self.range_slider)

        controls = QHBoxLayout()

        self.prev_btn = QPushButton("|<")
        self.prev_btn.clicked.connect(self.goto_start)
        controls.addWidget(self.prev_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_btn)

        self.next_btn = QPushButton(">")
        self.next_btn.clicked.connect(self.next_frame)
        controls.addWidget(self.next_btn)

        controls.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["depth", "conf"])
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        controls.addWidget(self.view_combo)

        controls.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.config.get("playback_speed", 10))
        self.fps_spin.setMaximumWidth(60)
        self.fps_spin.valueChanged.connect(self._on_speed_changed)
        controls.addWidget(self.fps_spin)

        controls.addStretch()

        controls.addWidget(QLabel("Session:"))
        self.session_combo = QComboBox()
        self.session_combo.currentTextChanged.connect(self.session_selected)
        controls.addWidget(self.session_combo)

        layout.addLayout(controls)

        return widget

    def _create_right_sidebar(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        camera_group = QGroupBox("Camera")
        camera_layout = QFormLayout(camera_group)
        camera_layout.addRow(
            "Focal:", QLabel(f"{self.config.get('focal_length_mm', 3.4)}mm")
        )
        camera_layout.addRow("Sensor:", QLabel(self.config.get("sensor_size", "1/6")))
        layout.addWidget(camera_group)

        conf_group = QGroupBox("Confidence")
        conf_layout = QFormLayout(conf_group)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 550)
        self.conf_slider.setValue(self.config.get("conf_threshold", 100))
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(0, 550)
        self.conf_spin.setValue(self.config.get("conf_threshold", 100))
        self.conf_spin.valueChanged.connect(self._on_conf_changed)
        conf_layout.addRow("Threshold:", self.conf_slider)
        conf_layout.addRow("", self.conf_spin)
        layout.addWidget(conf_group)

        bilateral_group = QGroupBox()
        self.bilateral_layout = QFormLayout(bilateral_group)
        self.bilateral_cb = QCheckBox("Bilateral Filter")
        self.bilateral_cb.setChecked(self.config.get("bilateral_enabled", False))
        self.bilateral_cb.toggled.connect(self._on_bilateral_toggled)
        self.bilateral_layout.addRow(self.bilateral_cb)
        self.bilateral_d = QSpinBox()
        self.bilateral_d.setRange(1, 21)
        self.bilateral_d.setSingleStep(2)
        self.bilateral_d.setValue(self.config.get("bilateral_d", 9))
        self.bilateral_d.valueChanged.connect(self._on_filter_changed)
        self.bilateral_layout.addRow("d:", self.bilateral_d)
        self.bilateral_sigma_color = QSpinBox()
        self.bilateral_sigma_color.setRange(1, 255)
        self.bilateral_sigma_color.setValue(
            self.config.get("bilateral_sigma_color", 200)
        )
        self.bilateral_sigma_color.valueChanged.connect(self._on_filter_changed)
        self.bilateral_layout.addRow("sigmaColor:", self.bilateral_sigma_color)
        self.bilateral_sigma_space = QSpinBox()
        self.bilateral_sigma_space.setRange(1, 100)
        self.bilateral_sigma_space.setValue(
            self.config.get("bilateral_sigma_space", 20)
        )
        self.bilateral_sigma_space.valueChanged.connect(self._on_filter_changed)
        self.bilateral_layout.addRow("sigmaSpace:", self.bilateral_sigma_space)
        self.bilateral_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.bilateral_rows = [
            self.bilateral_layout.rowCount() - 3,
            self.bilateral_layout.rowCount() - 2,
            self.bilateral_layout.rowCount() - 1,
        ]
        self._update_bilateral_params()
        layout.addWidget(bilateral_group)

        median_group = QGroupBox()
        self.median_layout = QFormLayout(median_group)
        self.median_cb = QCheckBox("Median Filter")
        self.median_cb.setChecked(self.config.get("median_enabled", False))
        self.median_cb.toggled.connect(self._on_median_toggled)
        self.median_layout.addRow(self.median_cb)
        self.median_k = QSpinBox()
        self.median_k.setRange(3, 15)
        self.median_k.setSingleStep(2)
        self.median_k.setValue(self.config.get("median_k", 5))
        self.median_k.valueChanged.connect(self._on_filter_changed)
        self.median_layout.addRow("k:", self.median_k)
        self.median_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._update_median_params()
        layout.addWidget(median_group)

        vo_group = QGroupBox("VO Settings")
        vo_layout = QFormLayout(vo_group)

        self.vo_detector_spin = QSpinBox()
        self.vo_detector_spin.setRange(1, 100)
        self.vo_detector_spin.setValue(self.config.get("detector_threshold", 20))
        self.vo_detector_spin.valueChanged.connect(self._on_vo_setting_changed)
        vo_layout.addRow("Detector:", self.vo_detector_spin)

        self.vo_rejection_spin = QSpinBox()
        self.vo_rejection_spin.setRange(1, 50)
        self.vo_rejection_spin.setValue(
            int(self.config.get("rejection_threshold", 8.0) * 10)
        )
        self.vo_rejection_spin.valueChanged.connect(self._on_vo_setting_changed)
        vo_layout.addRow("Rejection:", self.vo_rejection_spin)

        self.vo_stationary_spin = QSpinBox()
        self.vo_stationary_spin.setRange(1, 100)
        self.vo_stationary_spin.setValue(
            int(self.config.get("stationary_threshold", 0.5) * 100)
        )
        self.vo_stationary_spin.valueChanged.connect(self._on_vo_setting_changed)
        vo_layout.addRow("Stationary:", self.vo_stationary_spin)
        layout.addWidget(vo_group)

        clahe_group = QGroupBox("CLAHE")
        clahe_layout = QFormLayout(clahe_group)
        self.clahe_cb = QCheckBox("Enable CLAHE")
        self.clahe_cb.setChecked(self.config.get("clahe_enabled", False))
        self.clahe_cb.toggled.connect(self._on_clahe_toggled)
        clahe_layout.addRow("", self.clahe_cb)
        self.clahe_clip_spin = QSpinBox()
        self.clahe_clip_spin.setRange(1, 100)
        self.clahe_clip_spin.setValue(
            int(self.config.get("clahe_clip_limit", 2.0) * 10)
        )
        self.clahe_clip_spin.valueChanged.connect(self._on_clahe_changed)
        clahe_layout.addRow("Clip Limit:", self.clahe_clip_spin)
        self.clahe_grid_spin = QSpinBox()
        self.clahe_grid_spin.setRange(2, 16)
        self.clahe_grid_spin.setValue(self.config.get("clahe_grid_size", 8))
        self.clahe_grid_spin.valueChanged.connect(self._on_clahe_changed)
        clahe_layout.addRow("Grid Size:", self.clahe_grid_spin)
        self._update_clahe_params()
        layout.addWidget(clahe_group)

        layout.addStretch()

        return widget

    def _populate_sessions(self):
        data_dir = Path(__file__).parent / "data"
        if data_dir.exists():
            sessions = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            self.session_combo.addItems([""] + sessions)

    def _on_series_toggled(self):
        for key, cb in self.series_cbs.items():
            self.plot_curves[key].setVisible(cb.isChecked())
        self._update_graph()

    def _on_overlay_toggled(self):
        self.update_display()

    def _on_vo_enabled_changed(self, enabled):
        self.config["vo_enabled"] = enabled
        self._save_config()
        self.update_display()

    def _graph_mouse_move(self, ev):
        pos = ev.pos()
        vb = self.graph_widget.getViewBox()
        mouse_point = vb.mapSceneToView(pos)
        x = int(mouse_point.x())

        min_idx, max_idx = self.range_slider.value()
        if min_idx <= x <= max_idx and self.graph_data.get("tracked"):
            tooltip_lines = [f"Frame: {x}"]
            for key, curve in self.plot_curves.items():
                if curve.isVisible() and 0 <= x - min_idx < len(
                    self.graph_data.get(key, [])
                ):
                    value = self.graph_data[key][x - min_idx]
                    tooltip_lines.append(f"{key}: {value:.2f}")
            self.graph_widget.setToolTip("<br>".join(tooltip_lines))
        else:
            self.graph_widget.setToolTip("")

        ev.accept()

    def _on_range_changed(self):
        min_idx, max_idx = self.range_slider.value()
        if self.current_frame < min_idx:
            self.current_frame = min_idx
            self.slider.setValue(self.current_frame)
        elif self.current_frame > max_idx:
            self.current_frame = max_idx
            self.slider.setValue(self.current_frame)
        self._update_graph()

    def _on_view_changed(self, mode):
        self.config["view_mode"] = self.view_combo.currentText()
        self._save_config()
        self.update_display()

    def _on_conf_changed(self, value):
        self.config["conf_threshold"] = value
        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(value)
        self.conf_slider.blockSignals(False)
        self.conf_spin.blockSignals(True)
        self.conf_spin.setValue(value)
        self.conf_spin.blockSignals(False)
        self._save_config()
        self.update_display()

    def _on_speed_changed(self, value):
        self.config["playback_speed"] = value
        self._save_config()
        if self.playing:
            self.playback_timer.start(int(1000 / value))

    def _on_bilateral_toggled(self, checked):
        self.config["bilateral_enabled"] = checked
        self._save_config()
        self._update_bilateral_params()
        self.update_display()

    def _on_median_toggled(self, checked):
        self.config["median_enabled"] = checked
        self._save_config()
        self._update_median_params()
        self.update_display()

    def _on_filter_changed(self):
        self.config["bilateral_d"] = self.bilateral_d.value()
        self.config["bilateral_sigma_color"] = self.bilateral_sigma_color.value()
        self.config["bilateral_sigma_space"] = self.bilateral_sigma_space.value()
        self.config["median_k"] = self.median_k.value()
        self._save_config()
        self.update_display()

    def _update_bilateral_params(self):
        enabled = self.bilateral_cb.isChecked()
        for row in self.bilateral_rows:
            self.bilateral_layout.setRowVisible(row, enabled)

    def _update_median_params(self):
        enabled = self.median_cb.isChecked()
        self.median_layout.setRowVisible(1, enabled)

    def _on_vo_setting_changed(self):
        self.config["detector_threshold"] = self.vo_detector_spin.value()
        self.config["rejection_threshold"] = self.vo_rejection_spin.value() / 10.0
        self.config["stationary_threshold"] = self.vo_stationary_spin.value() / 100.0
        self._save_config()
        self.vo.update_config(self.config)
        self._recompute_vo()

    def _on_clahe_toggled(self, checked):
        self.config["clahe_enabled"] = checked
        self._save_config()
        self._update_clahe_params()
        self.vo.update_config(self.config)
        self._recompute_vo()

    def _on_clahe_changed(self):
        self.config["clahe_clip_limit"] = self.clahe_clip_spin.value() / 10.0
        self.config["clahe_grid_size"] = self.clahe_grid_spin.value()
        self._save_config()
        self.vo.update_config(self.config)
        self._recompute_vo()

    def _update_clahe_params(self):
        enabled = self.clahe_cb.isChecked()
        self.clahe_clip_spin.setEnabled(enabled)
        self.clahe_grid_spin.setEnabled(enabled)

    def session_selected(self, name):
        if not name:
            return
        data_dir = Path(__file__).parent / "data"
        self.session_dir = data_dir / name

        frames_dir = self.session_dir / "frames"
        frame_files = sorted(frames_dir.glob("frame_*.npz"))
        self.frames = [np.load(f) for f in frame_files]

        imu_path = self.session_dir / "imu_data.csv"
        if imu_path.exists():
            self.imu_data = np.genfromtxt(imu_path, delimiter=",", skip_header=1)

        self.current_frame = 0
        self.slider.setMaximum(len(self.frames) - 1)
        self.slider.setValue(0)
        self.range_slider.setRange(0, len(self.frames) - 1)
        self.range_slider.setValue(0, len(self.frames) - 1)

        self.view_combo.setCurrentText(self.config.get("view_mode", "depth"))

        self.vo.reset()
        self.vo_results = [None] * len(self.frames)
        for key in self.graph_data:
            self.graph_data[key] = [0] * len(self.frames)

        self.update_display()

    def _recompute_vo(self):
        if not self.frames:
            return
        self.vo.reset()
        self.vo_results = [None] * len(self.frames)
        for key in self.graph_data:
            self.graph_data[key] = [0] * len(self.frames)
        self.update_display()

    def slider_moved(self, value):
        self.current_frame = value
        self.update_display()

    def goto_start(self):
        self.current_frame = 0
        self.slider.setValue(0)

    def next_frame(self):
        min_idx, max_idx = self.range_slider.value()

        if self.current_frame < max_idx:
            self.current_frame += 1
            self.slider.setValue(self.current_frame)
        else:
            self.current_frame = min_idx
            self.slider.setValue(self.current_frame)

    def toggle_play(self):
        if self.playing:
            self.playback_timer.stop()
            self.play_btn.setText("Play")
            self.playing = False
        else:
            fps = self.fps_spin.value()
            self.playback_timer.start(int(1000 / fps))
            self.play_btn.setText("Pause")
            self.playing = True

    def eventFilter(self, obj, event):
        if obj == self.image_label and event.type() == QEvent.Type.MouseMove:
            if self.current_depth is not None and self.current_conf is not None:
                pixmap = self.image_label.pixmap()
                if pixmap:
                    scaled_w = pixmap.width()
                    scaled_h = pixmap.height()
                    orig_w = self.current_depth.shape[1]
                    orig_h = self.current_depth.shape[0]
                    x = int(event.position().x() * orig_w / scaled_w)
                    y = int(event.position().y() * orig_h / scaled_h)
                    if 0 <= x < orig_w and 0 <= y < orig_h:
                        depth_val = self.current_depth[y, x]
                        conf_val = self.current_conf[y, x]
                        self.image_label.setToolTip(
                            f"Depth: {depth_val:.2f}\nConf: {conf_val:.2f}"
                        )
        return super().eventFilter(obj, event)

    def _on_graph_mouse_moved(self, pos):
        if self.graph_widget is None:
            return
        try:
            mouse_point = self.graph_widget.plotItem.vb.mapSceneToView(pos)
        except Exception:
            return
        x = int(mouse_point.x())

        min_idx, max_idx = self.range_slider.value()
        if min_idx <= x <= max_idx and self.graph_data.get("tracked"):
            lines = [f"<div style='text-align: center'><b>Frame {x}</b><br>"]
            for key, curve in self.plot_curves.items():
                if curve.isVisible() and 0 <= x - min_idx < len(
                    self.graph_data.get(key, [])
                ):
                    value = self.graph_data[key][x - min_idx]
                    color = self.graph_colors.get(key, "#fff")
                    label = key.replace("_", " ").title()
                    lines.append(
                        f"<span style='color: {color}'>{label}: {value:.1f}</span><br>"
                    )
            lines.append("</div>")
            self.graph_tooltip.setHtml("".join(lines))
            self.graph_tooltip.setPos(x, mouse_point.y())
            self.graph_tooltip.show()
        else:
            self.graph_tooltip.hide()

    def _update_graph(self):
        if not self.graph_data["tracked"]:
            return

        min_idx, max_idx = self.range_slider.value()
        x = list(range(min_idx, max_idx + 1))

        for key, curve in self.plot_curves.items():
            data = self.graph_data[key][min_idx : max_idx + 1]
            data_arr = np.array(data)
            max_val = data_arr.max() if data_arr.size > 0 and data_arr.max() != 0 else 1
            normalized = data_arr / max_val * 100
            curve.setData(x, normalized)

        self.playhead_line.setValue(self.current_frame)

    def update_display(self):
        if not self.frames:
            return

        frame = self.frames[self.current_frame]
        view_mode = self.view_combo.currentText()
        conf_threshold = self.config.get("conf_threshold", 100)

        depth = frame["depth"]
        conf = frame["conf"]
        ts = frame["timestamp_ns"]

        if (
            self.vo_enabled_cb.isChecked()
            and self.vo_results[self.current_frame] is None
        ):
            result = self.vo.process(depth, conf, ts)
            self.vo_results[self.current_frame] = result

            for key in self.graph_data:
                self.graph_data[key][self.current_frame] = result.get(key, 0)
        else:
            result = self.vo_results[self.current_frame]

        self.current_depth = depth
        self.current_conf = conf

        masked_depth = np.where(conf >= conf_threshold, depth, 0)

        if view_mode == "depth":
            data = masked_depth.copy()
            if self.config.get("bilateral_enabled", False):
                d = self.config.get("bilateral_d", 9)
                sigma_color = self.config.get("bilateral_sigma_color", 200)
                sigma_space = self.config.get("bilateral_sigma_space", 20)
                data = cv2.bilateralFilter(
                    data.astype(np.float32), d, sigma_color, sigma_space
                )
            if self.config.get("median_enabled", False):
                k = self.config.get("median_k", 5)
                data = cv2.medianBlur(data.astype(np.float32), k)
            data = (
                np.clip(data * 255 / data.max(), 0, 255).astype(np.uint8)
                if data.max() > 0
                else data.astype(np.uint8)
            )
            data = cv2.applyColorMap(data, cv2.COLORMAP_VIRIDIS)
        else:
            data = conf.copy()
            data = (data * 255 / 545).astype(np.uint8)
            data = cv2.applyColorMap(data, cv2.COLORMAP_VIRIDIS)

        if self.show_tracked_cb.isChecked() or self.show_rejected_cb.isChecked():
            if self.current_frame < len(self.vo_results):
                result = self.vo_results[self.current_frame]

                if self.show_tracked_cb.isChecked():
                    for pt in result.get("tracked_pts", []):
                        x, y = int(pt[0]), int(pt[1])
                        if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
                            cv2.circle(data, (x, y), 3, (0, 255, 0), -1)

                if self.show_rejected_cb.isChecked():
                    for pt in result.get("rejected_pts", []):
                        x, y = int(pt[0]), int(pt[1])
                        if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
                            cv2.circle(data, (x, y), 3, (255, 0, 0), -1)

        h, w = data.shape[:2]
        bytes_per_line = data.strides[0]
        qimg = QImage(data.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        if self.current_frame < len(self.vo_results):
            result = self.vo_results[self.current_frame]
            if result:
                self.tracked_label.setText(str(result.get("tracked", 0)))
                self.rejected_label.setText(str(result.get("rejected", 0)))
                self.ratio_label.setText(f"{result.get('ratio', 0):.1f}%")
                self.vo_fps_label.setText(f"{result.get('fps', 0):.1f}")

        ts = frame["timestamp_ns"]
        info = f"Frame {self.current_frame + 1}/{len(self.frames)} | Timestamp: {ts}"
        if self.imu_data is not None:
            info += f" | IMU samples: {len(self.imu_data)}"

        self._update_graph()


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    viewer = DataViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
