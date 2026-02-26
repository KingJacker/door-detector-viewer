import json
import sys
from pathlib import Path

import numpy as np
import cv2
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSizePolicy,
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
    QDoubleSpinBox,
    QCheckBox,
    QFormLayout,
    QScrollArea,
    QTabWidget,
    QProgressBar,
)
from PySide6.QtCore import Qt, QEvent, QTimer, QThread
from PySide6.QtGui import QImage, QPixmap

from vo import VisualOdometry
from range_slider import RangeSlider
from ekf import ExtendedKalmanFilter
from loader import AsyncFrameLoader
from engine import ProcessingEngine
from worker import ProcessingWorker


CONFIG_PATH = Path(__file__).parent / "config.json"


class DataViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Door Detector Data Viewer")
        self.resize(1600, 900)

        self.session_dir = None
        self.imu_data = None
        self.current_frame = 0
        self.current_depth = None
        self.current_conf = None
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.next_frame)
        self.playing = False

        self.config = self._load_config()

        self.frame_loader = AsyncFrameLoader()
        self.engine = ProcessingEngine(self.config)

        self.vo_results = []

        # Apply saved EKF parameters to engine's EKF
        self.engine.update_config(self.config)

        self.imu_filter_alpha = self.config.get("imu_filter_alpha", 0.8)
        self.imu_filter_enabled = self.config.get("imu_filter_enabled", True)
        self.last_displayed_frame = -1

        # Top-down map state
        self.map_vo_positions = []  # list of (x, y)
        self.map_fused_positions = []  # list of (x, y)
        self.map_depth_points = []  # list of (x, y) — accumulated point cloud

        self.denoise_alpha = 0.3

        self.imu_filtered = {
            key: [] for key in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        }
        self.imu_prev = {
            key: 0.0
            for key in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        }
        self.imu_filter_enabled = False
        self.imu_filter_alpha = 0.8

        self.imu_interpolated = {
            key: [] for key in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        }

        self.graph_data = {
            "tracked": [],
            "rejected": [],
            "ratio": [],
            "pos_x": [],
            "pos_y": [],
            "pos_z": [],
            "pos_fused_x": [],
            "pos_fused_y": [],
            "pos_fused_z": [],
            "mean_depth": [],
            "mean_conf": [],
            "fps": [],
            "acc_x": [],
            "acc_y": [],
            "acc_z": [],
            "gyro_x": [],
            "gyro_y": [],
            "gyro_z": [],
            "acc_x_filt": [],
            "acc_y_filt": [],
            "acc_z_filt": [],
            "gyro_x_filt": [],
            "gyro_y_filt": [],
            "gyro_z_filt": [],
            "vo_score": [],
        }

        self.graph_colors = {
            "tracked": "#00ff00",
            "rejected": "#ff0000",
            "ratio": "#ffff00",
            "pos_x": "#ff00ff",
            "pos_y": "#00ffff",
            "pos_z": "#ff8800",
            "pos_fused_x": "#ff00aa",
            "pos_fused_y": "#00ffaa",
            "pos_fused_z": "#aaff00",
            "mean_depth": "#88ff88",
            "mean_conf": "#8888ff",
            "fps": "#ffffff",
            "acc_x": "#ff6666",
            "acc_y": "#66ff66",
            "acc_z": "#6666ff",
            "gyro_x": "#ffaaaa",
            "gyro_y": "#aaffaa",
            "gyro_z": "#aaaaff",
            "acc_x_filt": "#ff9999",
            "acc_y_filt": "#99ff99",
            "acc_z_filt": "#9999ff",
            "gyro_x_filt": "#ffcccc",
            "gyro_y_filt": "#ccffcc",
            "gyro_z_filt": "#ccccff",
            "vo_score": "#ffffff",
        }

        self._setup_ui()

    def _load_config(self):
        if CONFIG_PATH.exists():
            try:
                return json.loads(CONFIG_PATH.read_text())
            except Exception:
                pass
        return {
            "conf_threshold": 10,
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
            # EKF
            "ekf_enabled": False,
            "ekf_process_noise": 0.1,
            "ekf_measure_noise": 1.0,
            "fixed_height_enabled": False,
            "fixed_height": 1.2,
            "height_strength": 0.5,
            "max_speed": 2.0,
            "max_rotation_speed": 1.5,
            "lateral_damping": 0.1,
            "min_vo_confidence": 0.0,
            "low_conf_mode": "both",
            # IMU
            "imu_filter_enabled": True,
            "imu_filter_alpha": 0.8,
            "gyro_axis_map": "ZYX",
            # Denoise
            "denoise_alpha": 0.3,
            "vo_use_denoised": False,
            # Map
            "map_show_vo": True,
            "map_show_fused": True,
            "map_show_pts": False,
            "map_max_pts": 20000,
            "use_ekf_pose_for_points": True,
            "dense_point_cloud": False,
            "point_cloud_pixel_skip": 4,
            "constrain_max_range": True,
            "constrain_to_front": True,
            "max_range_mm": 4000,
            "camera_fov_deg": 70,
            # Camera parameter modes
            "use_fov_mode": False,
            "fov_x_deg": 70,
            "fov_y_deg": 55,
            "resolution_x": 240,
            "resolution_y": 180,
            "min_depth_mm": 100,
            # Graph series visibility
            "graph_series": {
                "tracked": True,
                "rejected": True,
                "ratio": True,
                "pos_x": False,
                "pos_y": False,
                "pos_z": False,
                "pos_fused_x": False,
                "pos_fused_y": False,
                "pos_fused_z": False,
                "mean_depth": False,
                "mean_conf": False,
                "fps": False,
                "vo_score": True,
                "acc_x": False,
                "acc_y": False,
                "acc_z": False,
                "gyro_x": False,
                "gyro_y": False,
                "gyro_z": False,
                "acc_x_filt": False,
                "acc_y_filt": False,
                "acc_z_filt": False,
                "gyro_x_filt": False,
                "gyro_y_filt": False,
                "gyro_z_filt": False,
            },
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

        center_widget = self._create_center_panel()
        left_sidebar = self._create_left_sidebar()
        right_sidebar = self._create_right_sidebar()

        splitter.addWidget(left_sidebar)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_sidebar)

        splitter.setSizes([250, 900, 300])

        self._populate_sessions()

    def _create_left_sidebar(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        camera_group = QGroupBox("Camera")
        camera_layout = QFormLayout(camera_group)

        # Focal length (configurable)
        self.focal_spin = QDoubleSpinBox()
        self.focal_spin.setRange(1.0, 10.0)
        self.focal_spin.setSingleStep(0.1)
        self.focal_spin.setValue(self.config.get("focal_length_mm", 3.4))
        self.focal_spin.setSuffix(" mm")
        self.focal_spin.valueChanged.connect(self._on_camera_params_changed)
        camera_layout.addRow("Focal:", self.focal_spin)

        # Sensor size (configurable)
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(["1/6", "1/4", "1/3", "1/2.3", "1/2", "2/3", "1"])
        self.sensor_combo.setCurrentText(self.config.get("sensor_size", "1/6"))
        self.sensor_combo.currentTextChanged.connect(self._on_camera_params_changed)
        camera_layout.addRow("Sensor:", self.sensor_combo)

        # Max range (configurable)
        self.max_range_spin = QSpinBox()
        self.max_range_spin.setRange(1000, 10000)
        self.max_range_spin.setSingleStep(500)
        self.max_range_spin.setValue(self.config.get("max_range_mm", 4000))
        self.max_range_spin.setSuffix(" mm")
        self.max_range_spin.setToolTip(
            "Maximum depth range for point cloud projection and FOV cone.\n"
            "Larger values show more distant points but may include noise."
        )
        self.max_range_spin.valueChanged.connect(self._on_max_range_changed)
        camera_layout.addRow("Max range:", self.max_range_spin)

        # Min depth / dead zone (configurable)
        self.min_depth_spin = QSpinBox()
        self.min_depth_spin.setRange(0, 1000)
        self.min_depth_spin.setSingleStep(10)
        self.min_depth_spin.setValue(self.config.get("min_depth_mm", 100))
        self.min_depth_spin.setSuffix(" mm")
        self.min_depth_spin.setToolTip(
            "Minimum depth threshold for point cloud projection.\n"
            "Points closer than this are ignored (shown as red dead zone in FOV cone)."
        )
        self.min_depth_spin.valueChanged.connect(self._on_min_depth_changed)
        camera_layout.addRow("Min depth:", self.min_depth_spin)

        # Mode toggle
        self.use_fov_mode_cb = QCheckBox("Use FOV mode")
        self.use_fov_mode_cb.setChecked(self.config.get("use_fov_mode", False))
        self.use_fov_mode_cb.setToolTip(
            "Toggle between physical camera params (focal length + sensor) "
            "and direct FOV + resolution input."
        )
        self.use_fov_mode_cb.toggled.connect(self._on_camera_mode_toggled)
        camera_layout.addRow(self.use_fov_mode_cb)

        # Physical mode widgets container
        self.physical_mode_widget = QWidget()
        physical_layout = QFormLayout(self.physical_mode_widget)
        physical_layout.setContentsMargins(0, 0, 0, 0)

        # Show computed FOV in physical mode
        self.computed_fov_x_label = QLabel("--")
        self.computed_fov_y_label = QLabel("--")
        physical_layout.addRow("FOV X:", self.computed_fov_x_label)
        physical_layout.addRow("FOV Y:", self.computed_fov_y_label)
        camera_layout.addRow(self.physical_mode_widget)

        # FOV mode widgets container
        self.fov_mode_widget = QWidget()
        fov_layout = QFormLayout(self.fov_mode_widget)
        fov_layout.setContentsMargins(0, 0, 0, 0)

        # FOV X
        self.fov_x_spin = QDoubleSpinBox()
        self.fov_x_spin.setRange(10.0, 180.0)
        self.fov_x_spin.setSingleStep(1.0)
        self.fov_x_spin.setDecimals(1)
        self.fov_x_spin.setValue(self.config.get("fov_x_deg", 70.0))
        self.fov_x_spin.setSuffix(" °")
        self.fov_x_spin.valueChanged.connect(self._on_fov_params_changed)
        fov_layout.addRow("FOV X:", self.fov_x_spin)

        # FOV Y
        self.fov_y_spin = QDoubleSpinBox()
        self.fov_y_spin.setRange(10.0, 180.0)
        self.fov_y_spin.setSingleStep(1.0)
        self.fov_y_spin.setDecimals(1)
        self.fov_y_spin.setValue(self.config.get("fov_y_deg", 55.0))
        self.fov_y_spin.setSuffix(" °")
        self.fov_y_spin.valueChanged.connect(self._on_fov_params_changed)
        fov_layout.addRow("FOV Y:", self.fov_y_spin)

        # Resolution X
        self.res_x_spin = QSpinBox()
        self.res_x_spin.setRange(100, 1000)
        self.res_x_spin.setSingleStep(10)
        self.res_x_spin.setValue(self.config.get("resolution_x", 240))
        self.res_x_spin.setSuffix(" px")
        self.res_x_spin.valueChanged.connect(self._on_fov_params_changed)
        fov_layout.addRow("Res X:", self.res_x_spin)

        # Resolution Y
        self.res_y_spin = QSpinBox()
        self.res_y_spin.setRange(100, 1000)
        self.res_y_spin.setSingleStep(10)
        self.res_y_spin.setValue(self.config.get("resolution_y", 180))
        self.res_y_spin.setSuffix(" px")
        self.res_y_spin.valueChanged.connect(self._on_fov_params_changed)
        fov_layout.addRow("Res Y:", self.res_y_spin)

        # Show computed focal length in FOV mode
        self.computed_focal_label = QLabel("--")
        fov_layout.addRow("Focal (px):", self.computed_focal_label)

        camera_layout.addRow(self.fov_mode_widget)

        # Set initial visibility
        self._update_camera_mode_visibility()

        layout.addWidget(camera_group)

        series_group = QGroupBox("Graph Series")
        series_layout = QVBoxLayout(series_group)

        self.series_cbs = {}
        self.series_labels = {}
        series_options = [
            ("tracked", "Tracked Points"),
            ("rejected", "Rejected Points"),
            ("ratio", "Ratio %"),
            ("pos_x", "Position X"),
            ("pos_y", "Position Y"),
            ("pos_z", "Position Z"),
            ("pos_fused_x", "Fused X"),
            ("pos_fused_y", "Fused Y"),
            ("pos_fused_z", "Fused Z"),
            ("mean_depth", "Mean Depth"),
            ("mean_conf", "Mean Conf"),
            ("fps", "FPS"),
            ("acc_x", "Acc X (raw)"),
            ("acc_y", "Acc Y (raw)"),
            ("acc_z", "Acc Z (raw)"),
            ("gyro_x", "Gyro X (raw)"),
            ("gyro_y", "Gyro Y (raw)"),
            ("gyro_z", "Gyro Z (raw)"),
            ("acc_x_filt", "Acc X (filt)"),
            ("acc_y_filt", "Acc Y (filt)"),
            ("acc_z_filt", "Acc Z (filt)"),
            ("gyro_x_filt", "Gyro X (filt)"),
            ("gyro_y_filt", "Gyro Y (filt)"),
            ("gyro_z_filt", "Gyro Z (filt)"),
            ("vo_score", "VO Score"),
        ]
        saved_series = self.config.get("graph_series", {})
        for key, label in series_options:
            color = self.graph_colors.get(key, "#fff")
            cb = QCheckBox(f"■ {label}")
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; }}")
            default_on = key in ("tracked", "rejected", "ratio", "vo_score")
            cb.setChecked(saved_series.get(key, default_on))
            cb.toggled.connect(self._on_series_toggled)
            self.series_cbs[key] = cb

            value_label = QLabel("0.0")
            value_label.setStyleSheet(f"QLabel {{ color: {color}; }}")
            value_label.setMinimumWidth(60)
            value_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.series_labels[key] = value_label

            row = QHBoxLayout()
            row.addWidget(cb)
            row.addWidget(value_label)
            series_layout.addLayout(row)

        layout.addWidget(series_group)
        layout.addStretch()

        # Sync initial curve visibility
        self._on_series_toggled()

        return widget

    def _create_center_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Tab widget: ToF View / Top-Down Map ---
        self.center_tabs = QTabWidget()
        self.center_tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Tab 1: ToF camera view
        tof_tab = QWidget()
        tof_layout = QVBoxLayout(tof_tab)
        tof_layout.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(480, 360)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        tof_layout.addWidget(self.image_label)

        self.pixel_info_label = QLabel("Hover over image for pixel info")
        self.pixel_info_label.setStyleSheet(
            "QLabel { color: #aaaaaa; font-family: monospace; padding: 2px 4px; }"
        )
        self.pixel_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        tof_layout.addWidget(self.pixel_info_label)

        self.center_tabs.addTab(tof_tab, "ToF View")

        # Tab 2: Top-down map
        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        map_layout.setContentsMargins(2, 2, 2, 2)

        self.map_widget = pg.PlotWidget()
        self.map_widget.setBackground("#1e1e1e")
        self.map_widget.setLabel("bottom", "X (m)")
        self.map_widget.setLabel("left", "Y (m)")
        self.map_widget.showGrid(x=True, y=True, alpha=0.3)
        self.map_widget.setAspectLocked(True)
        # Add padding so FOV cone and points can extend outside visible area
        self.map_widget.getViewBox().setDefaultPadding(0.1)
        self.map_widget.addLegend()

        # Path curves
        self.map_vo_curve = self.map_widget.plot(
            pen=pg.mkPen("#00ffff", width=2), name="VO path"
        )
        self.map_fused_curve = self.map_widget.plot(
            pen=pg.mkPen("#ff00aa", width=2), name="Fused path"
        )
        # Current position marker
        self.map_current_marker = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush("#ffffff"), pen=pg.mkPen(None)
        )
        self.map_widget.addItem(self.map_current_marker)
        # Depth point cloud scatter
        self.map_depth_scatter = pg.ScatterPlotItem(
            size=2, brush=pg.mkBrush(100, 200, 255, 80), pen=pg.mkPen(None)
        )
        self.map_widget.addItem(self.map_depth_scatter)

        # Camera FOV cone (outer - max range, gray with opacity)
        self.map_camera_fov = pg.PlotDataItem(
            pen=pg.mkPen((128, 128, 128, 100), width=1),
            fill=pg.mkBrush((128, 128, 128, 50)),
        )
        self.map_widget.addItem(self.map_camera_fov)

        # Camera dead zone cone (inner - min depth, red with opacity)
        self.map_dead_zone = pg.PlotDataItem(
            pen=pg.mkPen((255, 0, 0, 100), width=1),
            fill=pg.mkBrush((255, 0, 0, 50)),
        )
        self.map_widget.addItem(self.map_dead_zone)

        map_layout.addWidget(self.map_widget)

        # Map controls row
        map_ctrl = QHBoxLayout()
        self.map_show_vo_cb = QCheckBox("VO path")
        self.map_show_vo_cb.setChecked(self.config.get("map_show_vo", True))
        self.map_show_vo_cb.toggled.connect(self._on_map_options_changed)
        map_ctrl.addWidget(self.map_show_vo_cb)

        self.map_show_fused_cb = QCheckBox("Fused path")
        self.map_show_fused_cb.setChecked(self.config.get("map_show_fused", True))
        self.map_show_fused_cb.toggled.connect(self._on_map_options_changed)
        map_ctrl.addWidget(self.map_show_fused_cb)

        self.map_show_pts_cb = QCheckBox("Depth points")
        self.map_show_pts_cb.setChecked(self.config.get("map_show_pts", False))
        self.map_show_pts_cb.toggled.connect(self._on_map_options_changed)
        map_ctrl.addWidget(self.map_show_pts_cb)

        map_ctrl.addWidget(QLabel("Max pts:"))
        self.map_max_pts_spin = QSpinBox()
        self.map_max_pts_spin.setRange(100, 200000)
        self.map_max_pts_spin.setValue(self.config.get("map_max_pts", 20000))
        self.map_max_pts_spin.setSingleStep(1000)
        self.map_max_pts_spin.setToolTip(
            "Maximum number of depth points retained in the map point cloud."
        )
        self.map_max_pts_spin.valueChanged.connect(self._on_map_max_pts_changed)
        map_ctrl.addWidget(self.map_max_pts_spin)

        self.map_clear_btn = QPushButton("Clear map")
        self.map_clear_btn.clicked.connect(self._clear_map)
        map_ctrl.addWidget(self.map_clear_btn)
        map_ctrl.addStretch()
        map_layout.addLayout(map_ctrl)

        # Map options row 2 - point cloud settings
        map_opts2 = QHBoxLayout()

        self.use_ekf_pose_cb = QCheckBox("Use EKF pose")
        self.use_ekf_pose_cb.setChecked(
            self.config.get("use_ekf_pose_for_points", True)
        )
        self.use_ekf_pose_cb.setToolTip(
            "Use fused EKF pose for point projection (smoother).\n"
            "Uncheck to use raw VO pose (more jittery)."
        )
        self.use_ekf_pose_cb.toggled.connect(self._on_point_cloud_options_changed)
        map_opts2.addWidget(self.use_ekf_pose_cb)

        self.dense_cloud_cb = QCheckBox("Dense cloud")
        self.dense_cloud_cb.setChecked(self.config.get("dense_point_cloud", False))
        self.dense_cloud_cb.setToolTip(
            "Project all depth pixels (dense) vs tracked features only (sparse).\n"
            "Dense mode works without VO enabled."
        )
        self.dense_cloud_cb.toggled.connect(self._on_point_cloud_options_changed)
        map_opts2.addWidget(self.dense_cloud_cb)

        map_opts2.addWidget(QLabel("Skip:"))
        self.pixel_skip_spin = QSpinBox()
        self.pixel_skip_spin.setRange(1, 10)
        self.pixel_skip_spin.setValue(self.config.get("point_cloud_pixel_skip", 4))
        self.pixel_skip_spin.setToolTip(
            "Downsample dense cloud: 1=all pixels, 4=every 4th, etc."
        )
        self.pixel_skip_spin.valueChanged.connect(self._on_point_cloud_options_changed)
        map_opts2.addWidget(self.pixel_skip_spin)

        self.constrain_range_cb = QCheckBox("Max 4m")
        self.constrain_range_cb.setChecked(self.config.get("constrain_max_range", True))
        self.constrain_range_cb.setToolTip("Filter points beyond 4000mm sensor limit")
        self.constrain_range_cb.toggled.connect(self._on_point_cloud_options_changed)
        map_opts2.addWidget(self.constrain_range_cb)

        self.constrain_front_cb = QCheckBox("Front only")
        self.constrain_front_cb.setChecked(self.config.get("constrain_to_front", True))
        self.constrain_front_cb.setToolTip("Filter points behind camera (Z <= 0)")
        self.constrain_front_cb.toggled.connect(self._on_point_cloud_options_changed)
        map_opts2.addWidget(self.constrain_front_cb)

        map_opts2.addStretch()
        map_layout.addLayout(map_opts2)

        self.center_tabs.addTab(map_tab, "Top-Down Map")
        layout.addWidget(self.center_tabs, 3)

        # --- Frame slider (shared) ---
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        layout.addWidget(self.slider)

        # --- Time-series graph (shared) ---
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

        # --- Range slider ---
        self.range_slider = RangeSlider()
        self.range_slider.setRange(0, 100)
        self.range_slider.setValue(0, 100)
        self.range_slider.rangeChanged.connect(self._on_range_changed)
        layout.addWidget(self.range_slider)

        # --- Playback controls ---
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
        self.view_combo.addItems(["depth", "conf", "denoised"])
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        conf_group = QGroupBox("Confidence")
        conf_layout = QFormLayout(conf_group)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 200)
        self.conf_slider.setValue(self.config.get("conf_threshold", 10))
        self.conf_slider.setToolTip(
            "Minimum confidence (amplitude) for a valid pixel.\n"
            "Higher = fewer but more reliable pixels used in VO and display."
        )
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(0, 200)
        self.conf_spin.setValue(self.config.get("conf_threshold", 10))
        self.conf_spin.setToolTip(self.conf_slider.toolTip())
        self.conf_spin.valueChanged.connect(self._on_conf_changed)
        h = QHBoxLayout()
        h.addWidget(self.conf_slider)
        h.addWidget(self.conf_spin)
        conf_layout.addRow("Threshold:", h)
        layout.addWidget(conf_group)

        vo_group = QGroupBox("VO Settings")
        vo_layout = QFormLayout(vo_group)

        self.vo_enabled_cb = QCheckBox("Tracking")
        self.vo_enabled_cb.setChecked(True)
        self.vo_enabled_cb.toggled.connect(self._on_vo_enabled_changed)
        vo_layout.addRow(self.vo_enabled_cb)

        self.show_tracked_cb = QCheckBox("Show tracked points")
        self.show_tracked_cb.setChecked(False)
        self.show_tracked_cb.toggled.connect(self._on_overlay_toggled)
        vo_layout.addRow(self.show_tracked_cb)

        self.show_rejected_cb = QCheckBox("Show rejected points")
        self.show_rejected_cb.setChecked(False)
        self.show_rejected_cb.toggled.connect(self._on_overlay_toggled)
        vo_layout.addRow(self.show_rejected_cb)

        self.vo_detector_spin = QSpinBox()
        self.vo_detector_spin.setRange(1, 100)
        self.vo_detector_spin.setValue(int(self.config.get("detector_threshold", 20)))
        self.vo_detector_spin.setToolTip(
            "FAST corner detector threshold.\nLower = more features detected (slower)."
        )
        self.vo_detector_spin.valueChanged.connect(self._on_vo_setting_changed)
        vo_layout.addRow("Detector:", self.vo_detector_spin)

        self.vo_rejection_spin = QDoubleSpinBox()
        self.vo_rejection_spin.setRange(0.1, 20.0)
        self.vo_rejection_spin.setSingleStep(0.1)
        self.vo_rejection_spin.setDecimals(1)
        self.vo_rejection_spin.setValue(self.config.get("rejection_threshold", 8.0))
        self.vo_rejection_spin.setToolTip(
            "Max optical flow error in pixels.\nLower = stricter point filtering."
        )
        self.vo_rejection_spin.valueChanged.connect(self._on_vo_setting_changed)
        vo_layout.addRow("Rejection (px):", self.vo_rejection_spin)

        self.vo_stationary_spin = QDoubleSpinBox()
        self.vo_stationary_spin.setRange(0.0, 10.0)
        self.vo_stationary_spin.setSingleStep(0.1)
        self.vo_stationary_spin.setDecimals(2)
        self.vo_stationary_spin.setValue(self.config.get("stationary_threshold", 0.5))
        self.vo_stationary_spin.setToolTip(
            "Skip pose update when mean optical flow is below this (pixels).\n"
            "Prevents drift when the camera is stationary."
        )
        self.vo_stationary_spin.valueChanged.connect(self._on_vo_setting_changed)
        vo_layout.addRow("Stationary (px):", self.vo_stationary_spin)

        self.process_btn = QPushButton("Process Selected Range")
        self.process_btn.setToolTip(
            "Process the selected frame range in the background."
        )
        self.process_btn.clicked.connect(self._on_process_range_clicked)
        vo_layout.addRow(self.process_btn)

        self.vo_layout = vo_layout
        self.vo_rows = list(range(1, vo_layout.rowCount()))
        layout.addWidget(vo_group)

        clahe_group = QGroupBox("CLAHE")
        clahe_layout = QFormLayout(clahe_group)
        self.clahe_cb = QCheckBox("Enable CLAHE")
        self.clahe_cb.setChecked(self.config.get("clahe_enabled", False))
        self.clahe_cb.toggled.connect(self._on_clahe_toggled)
        clahe_layout.addRow(self.clahe_cb)
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
        self.clahe_layout = clahe_layout
        self.clahe_rows = list(range(1, clahe_layout.rowCount()))
        self._update_clahe_params()
        layout.addWidget(clahe_group)

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

        denoise_group = QGroupBox("Temporal Denoise")
        denoise_layout = QFormLayout(denoise_group)
        denoise_layout.addRow(
            QLabel(
                "Select 'denoised' view + enable EKF.\n"
                "Motion-compensated temporal filter."
            )
        )
        self.denoise_alpha = self.config.get("denoise_alpha", 0.3)
        self.denoise_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_alpha_slider.setRange(5, 100)
        self.denoise_alpha_slider.setValue(int(self.denoise_alpha * 100))
        self.denoise_alpha_slider.setToolTip(
            "Temporal blend alpha: 1.0 = no blending (current frame only),\n"
            "0.05 = heavy averaging (more smoothing, more lag)"
        )
        self.denoise_alpha_slider.valueChanged.connect(
            lambda v: self._on_denoise_alpha_changed(v / 100)
        )
        self.denoise_alpha_spin = QDoubleSpinBox()
        self.denoise_alpha_spin.setRange(0.05, 1.0)
        self.denoise_alpha_spin.setSingleStep(0.05)
        self.denoise_alpha_spin.setDecimals(2)
        self.denoise_alpha_spin.setValue(
            self.denoise_alpha
        )  # already set from config above
        self.denoise_alpha_spin.setToolTip(self.denoise_alpha_slider.toolTip())
        self.denoise_alpha_spin.valueChanged.connect(
            lambda v: self._on_denoise_alpha_changed(v)
        )
        h = QHBoxLayout()
        h.addWidget(self.denoise_alpha_slider)
        h.addWidget(self.denoise_alpha_spin)
        denoise_layout.addRow("Alpha (1=raw):", h)

        self.vo_use_denoised_cb = QCheckBox("Use denoised depth for VO")
        self.vo_use_denoised_cb.setChecked(self.config.get("vo_use_denoised", False))
        self.vo_use_denoised_cb.setToolTip(
            "Feed the temporally denoised depth into the VO pipeline.\n"
            "Reduces optical flow noise and improves scale recovery.\n"
            "Requires EKF to be enabled. Triggers a full VO recompute."
        )
        self.vo_use_denoised_cb.toggled.connect(self._on_vo_use_denoised_changed)
        denoise_layout.addRow(self.vo_use_denoised_cb)
        layout.addWidget(denoise_group)

        ekf_group = QGroupBox("EKF Settings")
        ekf_layout = QFormLayout(ekf_group)

        self.ekf_enabled_cb = QCheckBox("Enable EKF")
        self.ekf_enabled_cb.setChecked(self.config.get("ekf_enabled", False))
        self.ekf_enabled_cb.toggled.connect(self._on_ekf_enabled_changed)
        ekf_layout.addRow(self.ekf_enabled_cb)

        pn = self.config.get("ekf_process_noise", 0.1)
        self.ekf_process_noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.ekf_process_noise_slider.setRange(1, 100)
        self.ekf_process_noise_slider.setValue(int(pn * 100))
        self.ekf_process_noise_slider.setToolTip(
            "Process noise (Q) - higher = more trust in IMU"
        )
        self.ekf_process_noise_slider.valueChanged.connect(
            lambda v: self._on_ekf_param_changed("process_noise", v / 100)
        )
        self.ekf_process_noise_spin = QDoubleSpinBox()
        self.ekf_process_noise_spin.setRange(0.01, 1.0)
        self.ekf_process_noise_spin.setValue(pn)
        self.ekf_process_noise_spin.setSingleStep(0.01)
        self.ekf_process_noise_spin.valueChanged.connect(
            lambda v: self._on_ekf_param_changed("process_noise", v)
        )
        h = QHBoxLayout()
        h.addWidget(self.ekf_process_noise_slider)
        h.addWidget(self.ekf_process_noise_spin)
        ekf_layout.addRow("Process Noise:", h)

        mn = self.config.get("ekf_measure_noise", 1.0)
        self.ekf_measure_noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.ekf_measure_noise_slider.setRange(1, 100)
        self.ekf_measure_noise_slider.setValue(int(mn * 10))
        self.ekf_measure_noise_slider.setToolTip(
            "Measurement noise (R) - higher = less trust in VO"
        )
        self.ekf_measure_noise_slider.valueChanged.connect(
            lambda v: self._on_ekf_param_changed("measure_noise", v / 10)
        )
        self.ekf_measure_noise_spin = QDoubleSpinBox()
        self.ekf_measure_noise_spin.setRange(0.1, 10.0)
        self.ekf_measure_noise_spin.setValue(mn)
        self.ekf_measure_noise_spin.setSingleStep(0.1)
        self.ekf_measure_noise_spin.valueChanged.connect(
            lambda v: self._on_ekf_param_changed("measure_noise", v)
        )
        h = QHBoxLayout()
        h.addWidget(self.ekf_measure_noise_slider)
        h.addWidget(self.ekf_measure_noise_spin)
        ekf_layout.addRow("Meas. Noise:", h)

        self.fixed_height_cb = QCheckBox("Fixed Height Constraint")
        self.fixed_height_cb.setChecked(self.config.get("fixed_height_enabled", False))
        self.fixed_height_cb.toggled.connect(self._on_fixed_height_changed)
        ekf_layout.addRow(self.fixed_height_cb)

        fh = self.config.get("fixed_height", 1.2)
        self.fixed_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.fixed_height_slider.setRange(50, 250)
        self.fixed_height_slider.setValue(int(fh * 100))
        self.fixed_height_slider.setToolTip("Fixed height in meters (slider ÷ 100)")
        self.fixed_height_slider.valueChanged.connect(
            lambda v: self._on_height_param_changed(v / 100)
        )
        self.fixed_height_spin = QDoubleSpinBox()
        self.fixed_height_spin.setRange(0.5, 2.5)
        self.fixed_height_spin.setValue(fh)
        self.fixed_height_spin.setSingleStep(0.05)
        self.fixed_height_spin.setSuffix(" m")
        self.fixed_height_spin.valueChanged.connect(
            lambda v: self._on_height_param_changed(v)
        )
        h = QHBoxLayout()
        h.addWidget(self.fixed_height_slider)
        h.addWidget(self.fixed_height_spin)
        ekf_layout.addRow("Height:", h)

        hs = self.config.get("height_strength", 0.5)
        self.height_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.height_strength_slider.setRange(0, 100)
        self.height_strength_slider.setValue(int(hs * 100))
        self.height_strength_slider.setToolTip("Constraint strength (0=free, 1=locked)")
        self.height_strength_slider.valueChanged.connect(
            lambda v: self._on_height_strength_changed(v / 100)
        )
        self.height_strength_spin = QDoubleSpinBox()
        self.height_strength_spin.setRange(0.0, 1.0)
        self.height_strength_spin.setValue(hs)
        self.height_strength_spin.setSingleStep(0.05)
        self.height_strength_spin.valueChanged.connect(
            lambda v: self._on_height_strength_changed(v)
        )
        h = QHBoxLayout()
        h.addWidget(self.height_strength_slider)
        h.addWidget(self.height_strength_spin)
        ekf_layout.addRow("Height Strength:", h)

        ms = self.config.get("max_speed", 2.0)
        self.max_speed_spin = QDoubleSpinBox()
        self.max_speed_spin.setRange(0.0, 10.0)
        self.max_speed_spin.setSingleStep(0.1)
        self.max_speed_spin.setDecimals(1)
        self.max_speed_spin.setValue(ms)
        self.max_speed_spin.setSuffix(" m/s")
        self.max_speed_spin.setToolTip(
            "Maximum allowed speed.\n"
            "Clamps velocity after each EKF step and gates VO updates.\n"
            "Set to 0 to disable."
        )
        self.max_speed_spin.valueChanged.connect(self._on_max_speed_changed)
        ekf_layout.addRow("Max Speed:", self.max_speed_spin)

        mrs = self.config.get("max_rotation_speed", 1.5)
        self.max_rot_speed_spin = QDoubleSpinBox()
        self.max_rot_speed_spin.setRange(0.0, 5.0)
        self.max_rot_speed_spin.setSingleStep(0.1)
        self.max_rot_speed_spin.setDecimals(1)
        self.max_rot_speed_spin.setValue(mrs)
        self.max_rot_speed_spin.setSuffix(" rad/s")
        self.max_rot_speed_spin.setToolTip(
            "Maximum allowed rotation speed.\n"
            "Clamps angular velocity before EKF quaternion integration.\n"
            "~1.5 rad/s (86°/s) is reasonable for electric wheelchairs.\n"
            "Set to 0 to disable."
        )
        self.max_rot_speed_spin.valueChanged.connect(
            self._on_max_rotation_speed_changed
        )
        ekf_layout.addRow("Max Rot Speed:", self.max_rot_speed_spin)

        ld = self.config.get("lateral_damping", 0.1)
        self.lateral_damping_spin = QDoubleSpinBox()
        self.lateral_damping_spin.setRange(0.0, 1.0)
        self.lateral_damping_spin.setSingleStep(0.05)
        self.lateral_damping_spin.setDecimals(2)
        self.lateral_damping_spin.setValue(ld)
        self.lateral_damping_spin.setToolTip(
            "Lateral velocity damping factor.\n"
            "0 = no sideways motion (strict non-holonomic constraint).\n"
            "1 = unconstrained.\n"
            "Low values (~0.05-0.2) suit wheelchair motion."
        )
        self.lateral_damping_spin.valueChanged.connect(self._on_lateral_damping_changed)
        ekf_layout.addRow("Lateral Damping:", self.lateral_damping_spin)

        mvc = self.config.get("min_vo_confidence", 0.0)
        self.min_vo_conf_spin = QDoubleSpinBox()
        self.min_vo_conf_spin.setRange(0.0, 1.0)
        self.min_vo_conf_spin.setSingleStep(0.05)
        self.min_vo_conf_spin.setDecimals(2)
        self.min_vo_conf_spin.setValue(mvc)
        self.min_vo_conf_spin.setToolTip(
            "Minimum VO confidence score to accept a measurement update.\n"
            "Below this, the low-confidence mode strategy is applied."
        )
        self.min_vo_conf_spin.valueChanged.connect(self._on_min_vo_conf_changed)
        ekf_layout.addRow("Min VO Conf:", self.min_vo_conf_spin)

        self.low_conf_mode_combo = QComboBox()
        self.low_conf_mode_combo.addItems(
            [
                "Freeze VO (keep last pos)",
                "IMU only (EKF predict)",
                "Both (VO frozen + IMU)",
            ]
        )
        mode_map = {
            "freeze_vo": "Freeze VO (keep last pos)",
            "imu_only": "IMU only (EKF predict)",
            "both": "Both (VO frozen + IMU)",
        }
        self.low_conf_mode_combo.setCurrentText(
            mode_map.get(
                self.config.get("low_conf_mode", "both"), "Both (VO frozen + IMU)"
            )
        )
        self.low_conf_mode_combo.setToolTip(
            "What to do when VO confidence is below the threshold:\n"
            "  Freeze VO: keep the last valid VO position\n"
            "  IMU only: let EKF predict-only via IMU (will drift)\n"
            "  Both: freeze VO path AND use IMU-only for fused path"
        )
        self.low_conf_mode_combo.currentTextChanged.connect(
            self._on_low_conf_mode_changed
        )
        ekf_layout.addRow("Low Conf Mode:", self.low_conf_mode_combo)

        self.ekf_layout = ekf_layout
        self.ekf_rows = list(range(1, ekf_layout.rowCount()))
        self.fixed_height_rows = [4, 5]
        layout.addWidget(ekf_group)

        imu_group = QGroupBox("IMU Filter")
        imu_layout = QFormLayout(imu_group)

        self.imu_filter_cb = QCheckBox("Enable Low-Pass Filter")
        self.imu_filter_cb.setChecked(self.config.get("imu_filter_enabled", True))
        self.imu_filter_cb.toggled.connect(self._on_imu_filter_changed)
        imu_layout.addRow(self.imu_filter_cb)

        self.imu_mapping_combo = QComboBox()
        self.imu_mapping_combo.addItems(["XYZ (Standard)", "ZYX (Swap X\u2194Z)", "XZY", "YXZ", "YZX", "ZXY"])
        mapping_labels = {
            (0, 1, 2): "XYZ (Standard)",
            (2, 1, 0): "ZYX (Swap X\u2194Z)",
            (0, 2, 1): "XZY",
            (1, 0, 2): "YXZ",
            (1, 2, 0): "YZX",
            (2, 0, 1): "ZXY",
        }
        saved_m = tuple(self.config.get("imu_axis_mapping", [2, 1, 0])) # Default to ZYX for this project
        self.imu_mapping_combo.setCurrentText(mapping_labels.get(saved_m, "ZYX (Swap X\u2194Z)"))
        self.imu_mapping_combo.currentTextChanged.connect(self._on_imu_mapping_changed)
        imu_layout.addRow("Axis Mapping:", self.imu_mapping_combo)

        # Axis Signs UI
        h_signs = QHBoxLayout()
        self.sign_x_cb = QCheckBox("X")
        self.sign_y_cb = QCheckBox("Y")
        self.sign_z_cb = QCheckBox("Z")
        signs = self.config.get("imu_axis_signs", [1, 1, 1])
        self.sign_x_cb.setChecked(signs[0] > 0)
        self.sign_y_cb.setChecked(signs[1] > 0)
        self.sign_z_cb.setChecked(signs[2] > 0)
        for cb in [self.sign_x_cb, self.sign_y_cb, self.sign_z_cb]:
            cb.toggled.connect(self._on_imu_mapping_changed)
            h_signs.addWidget(cb)
        imu_layout.addRow("Axis Directions:", h_signs)

        self.log_ekf_cb = QCheckBox("Log EKF data to CSV")
        self.log_ekf_cb.setChecked(self.config.get("log_ekf", True))
        self.log_ekf_cb.setToolTip("Save raw IMU, fused pose and orientation to 'ekf_debug.log' in session folder.")
        self.log_ekf_cb.toggled.connect(self._on_log_ekf_toggled)
        imu_layout.addRow(self.log_ekf_cb)

        ia = self.config.get("imu_filter_alpha", 0.8)
        self.imu_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.imu_alpha_slider.setRange(0, 100)
        self.imu_alpha_slider.setValue(int(ia * 100))
        self.imu_alpha_slider.setToolTip(
            "Low-pass alpha: 1.0 = no filtering (raw), 0.0 = maximum smoothing"
        )
        self.imu_alpha_slider.valueChanged.connect(
            lambda v: self._on_imu_alpha_changed(v / 100)
        )
        self.imu_alpha_spin = QDoubleSpinBox()
        self.imu_alpha_spin.setRange(0.0, 1.0)
        self.imu_alpha_spin.setValue(ia)
        self.imu_alpha_spin.setSingleStep(0.05)
        self.imu_alpha_spin.setToolTip(
            "Low-pass alpha: 1.0 = no filtering (raw), 0.0 = maximum smoothing"
        )
        self.imu_alpha_spin.valueChanged.connect(
            lambda v: self._on_imu_alpha_changed(v)
        )
        h = QHBoxLayout()
        h.addWidget(self.imu_alpha_slider)
        h.addWidget(self.imu_alpha_spin)
        imu_layout.addRow("Alpha (1=raw):", h)

        self.imu_layout = imu_layout
        self.imu_rows = list(range(1, imu_layout.rowCount()))

        self._on_vo_enabled_changed(self.vo_enabled_cb.isChecked())
        self._on_ekf_enabled_changed(self.ekf_enabled_cb.isChecked())
        if self.ekf_enabled_cb.isChecked():
            self._on_fixed_height_changed(self.fixed_height_cb.isChecked())
        self._on_imu_filter_changed(self.imu_filter_cb.isChecked())
        self._on_clahe_toggled(self.clahe_cb.isChecked())
        self._on_map_options_changed()

        layout.addWidget(imu_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        scroll.setWidget(widget)

        return scroll

    def _populate_sessions(self):
        data_dir = Path(__file__).parent / "data"
        if not data_dir.exists():
            return
        sessions = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        self.session_combo.addItems([""] + sessions)

        # Restore the last active session without triggering a double-load
        last = self.config.get("last_session", "")
        if last and last in sessions:
            self.session_combo.blockSignals(True)
            self.session_combo.setCurrentText(last)
            self.session_combo.blockSignals(False)
            self.session_selected(last)

    def _on_series_toggled(self):
        series_state = {}
        for key, cb in self.series_cbs.items():
            checked = cb.isChecked()
            self.plot_curves[key].setVisible(checked)
            series_state[key] = checked
        self.config["graph_series"] = series_state
        self._save_config()
        self._update_graph()

    def _on_overlay_toggled(self):
        self.update_display()

    def _on_vo_enabled_changed(self, enabled):
        self.config["vo_enabled"] = enabled
        self._save_config()
        for row in self.vo_rows:
            self.vo_layout.setRowVisible(row, enabled)
        self.update_display()

    def _on_ekf_enabled_changed(self, enabled):
        self.config["ekf_enabled"] = enabled
        self._save_config()
        for row in self.ekf_rows:
            self.ekf_layout.setRowVisible(row, enabled)
        if not enabled:
            self.fixed_height_cb.setChecked(False)
            self._on_fixed_height_changed(False)
        if enabled:
            self.engine.reset(self.imu_data)
        self.update_display()

    def _on_ekf_param_changed(self, param, value):
        if param == "process_noise":
            self.engine.ekf.process_noise = value
            self.config["ekf_process_noise"] = value
            self.ekf_process_noise_slider.blockSignals(True)
            self.ekf_process_noise_slider.setValue(int(value * 100))
            self.ekf_process_noise_slider.blockSignals(False)
            self.ekf_process_noise_spin.blockSignals(True)
            self.ekf_process_noise_spin.setValue(value)
            self.ekf_process_noise_spin.blockSignals(False)
        elif param == "measure_noise":
            self.engine.ekf.measurement_noise = value
            self.config["ekf_measure_noise"] = value
            self.ekf_measure_noise_slider.blockSignals(True)
            self.ekf_measure_noise_slider.setValue(int(value * 10))
            self.ekf_measure_noise_slider.blockSignals(False)
            self.ekf_measure_noise_spin.blockSignals(True)
            self.ekf_measure_noise_spin.setValue(value)
            self.ekf_measure_noise_spin.blockSignals(False)
        self._save_config()
        self.engine.update_config(self.config)
        self.engine.reset(self.imu_data)
        self.update_display()

    def _on_fixed_height_changed(self, enabled):
        self.config["fixed_height_enabled"] = enabled
        self.engine.update_config(self.config)
        self._save_config()
        for row in self.fixed_height_rows:
            self.ekf_layout.setRowVisible(row, enabled)
        self.update_display()

    def _on_height_param_changed(self, height_meters):
        self.config["fixed_height"] = height_meters
        self.engine.update_config(self.config)
        self._save_config()
        self.fixed_height_slider.blockSignals(True)
        self.fixed_height_slider.setValue(int(height_meters * 100))
        self.fixed_height_slider.blockSignals(False)
        self.fixed_height_spin.blockSignals(True)
        self.fixed_height_spin.setValue(height_meters)
        self.fixed_height_spin.blockSignals(False)
        self.update_display()

    def _on_height_strength_changed(self, strength):
        self.config["height_strength"] = strength
        self.engine.update_config(self.config)
        self._save_config()
        self.height_strength_slider.blockSignals(True)
        self.height_strength_slider.setValue(int(strength * 100))
        self.height_strength_slider.blockSignals(False)
        self.height_strength_spin.blockSignals(True)
        self.height_strength_spin.setValue(strength)
        self.height_strength_spin.blockSignals(False)
        self.update_display()

    def _on_max_speed_changed(self, value):
        self.config["max_speed"] = value
        self.engine.update_config(self.config)
        self._save_config()
        self.update_display()

    def _on_max_rotation_speed_changed(self, value):
        self.config["max_rotation_speed"] = value
        self.engine.update_config(self.config)
        self._save_config()
        self.update_display()

    def _on_lateral_damping_changed(self, value):
        self.config["lateral_damping"] = value
        self.engine.update_config(self.config)
        self._save_config()
        self.update_display()

    def _on_min_vo_conf_changed(self, value):
        self.config["min_vo_confidence"] = value
        self._save_config()
        self.update_display()

    def _on_low_conf_mode_changed(self, text):
        mode_map = {
            "Freeze VO (keep last pos)": "freeze_vo",
            "IMU only (EKF predict)": "imu_only",
            "Both (VO frozen + IMU)": "both",
        }
        self.config["low_conf_mode"] = mode_map.get(text, "both")
        self._save_config()
        self.update_display()

    def _on_imu_mapping_changed(self):
        mapping_vals = {
            "XYZ (Standard)": [0, 1, 2],
            "ZYX (Swap X\u2194Z)": [2, 1, 0],
            "XZY": [0, 2, 1],
            "YXZ": [1, 0, 2],
            "YZX": [1, 2, 0],
            "ZXY": [2, 0, 1],
        }
        self.config["imu_axis_mapping"] = mapping_vals.get(self.imu_mapping_combo.currentText(), [0, 1, 2])
        self.config["imu_axis_signs"] = [
            1 if self.sign_x_cb.isChecked() else -1,
            1 if self.sign_y_cb.isChecked() else -1,
            1 if self.sign_z_cb.isChecked() else -1,
        ]
        self._save_config()
        self.engine.update_config(self.config)
        self.engine.reset(self.imu_data)
        self.update_display()

    def _on_log_ekf_toggled(self, checked):
        self.config["log_ekf"] = checked
        self._save_config()
        if checked and self.session_dir:
            self.engine.start_logging(self.session_dir / "ekf_debug.log")
        else:
            self.engine.stop_logging()

    def _on_vo_use_denoised_changed(self, checked):
        self.config["vo_use_denoised"] = checked
        self._save_config()
        # VO must recompute since its input changes
        self._recompute_vo()

    def _on_denoise_alpha_changed(self, alpha):
        self.denoise_alpha = alpha
        self.config["denoise_alpha"] = alpha
        self._save_config()
        self.denoise_alpha_slider.blockSignals(True)
        self.denoise_alpha_slider.setValue(int(alpha * 100))
        self.denoise_alpha_slider.blockSignals(False)
        self.denoise_alpha_spin.blockSignals(True)
        self.denoise_alpha_spin.setValue(alpha)
        self.denoise_alpha_spin.blockSignals(False)
        # Reset denoised buffer so it rebuilds with new alpha
        self.denoised_depth = None
        self.prev_ekf_pos = None
        self.prev_ekf_R = None
        self.update_display()

    def _on_imu_filter_changed(self, enabled):
        self.imu_filter_enabled = enabled
        self.config["imu_filter_enabled"] = enabled
        self._save_config()
        for row in self.imu_rows:
            self.imu_layout.setRowVisible(row, enabled)
        if self.frame_loader.frame_paths:
            self._recompute_imu_filter()
        self.update_display()

    def _on_imu_alpha_changed(self, alpha):
        self.imu_filter_alpha = alpha
        self.config["imu_filter_alpha"] = alpha
        self._save_config()
        self.imu_alpha_slider.blockSignals(True)
        self.imu_alpha_slider.setValue(int(alpha * 100))
        self.imu_alpha_slider.blockSignals(False)
        self.imu_alpha_spin.blockSignals(True)
        self.imu_alpha_spin.setValue(alpha)
        self.imu_alpha_spin.blockSignals(False)
        if self.frame_loader.frame_paths:
            self._recompute_imu_filter()
        self.update_display()

    def _recompute_imu_filter(self):
        alpha = self.imu_filter_alpha
        keys = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        num_frames = len(self.frame_loader.frame_paths)
        for key in keys:
            raw = self.imu_interpolated[key]
            filtered = [0.0] * num_frames
            prev = raw[0] if num_frames > 0 else 0.0
            for i in range(num_frames):
                val = alpha * raw[i] + (1 - alpha) * prev
                filtered[i] = val
                prev = val
            self.imu_filtered[key] = filtered
            # Mirror into graph_data so filtered series are visible in the graph
            self.graph_data[key + "_filt"] = list(filtered)

    def _on_map_options_changed(self):
        self.map_vo_curve.setVisible(self.map_show_vo_cb.isChecked())
        self.map_fused_curve.setVisible(self.map_show_fused_cb.isChecked())
        self.map_depth_scatter.setVisible(self.map_show_pts_cb.isChecked())
        self.config["map_show_vo"] = self.map_show_vo_cb.isChecked()
        self.config["map_show_fused"] = self.map_show_fused_cb.isChecked()
        self.config["map_show_pts"] = self.map_show_pts_cb.isChecked()
        self._save_config()

    def _on_map_max_pts_changed(self, value):
        self.config["map_max_pts"] = value
        self._save_config()

    def _on_point_cloud_options_changed(self, _=None):
        """Handle changes to point cloud display options."""
        self.config["use_ekf_pose_for_points"] = self.use_ekf_pose_cb.isChecked()
        self.config["dense_point_cloud"] = self.dense_cloud_cb.isChecked()
        self.config["point_cloud_pixel_skip"] = self.pixel_skip_spin.value()
        self.config["constrain_max_range"] = self.constrain_range_cb.isChecked()
        self.config["constrain_to_front"] = self.constrain_front_cb.isChecked()
        self._save_config()
        # Trigger update to show changes
        self.update_display()

    def _clear_paths(self):
        """Clear path curves only — keeps the accumulated depth point cloud."""
        self.map_vo_positions.clear()
        self.map_fused_positions.clear()
        self.map_vo_curve.setData([], [])
        self.map_fused_curve.setData([], [])
        self.map_current_marker.setData([], [])

    def _clear_map(self):
        """Clear everything including the depth point cloud."""
        self._clear_paths()
        self.map_depth_points.clear()
        self.map_depth_scatter.setData([], [])
        self.map_camera_fov.setData([], [])
        self.map_dead_zone.setData([], [])

    def _update_camera_fov(self, cx, cy, R_w=None):
        """Update the camera FOV cone visualization.

        Draws a triangular cone representing the camera's field of view in the
        top-down map view. The cone extends from the camera position forward
        along the camera's viewing direction to the configured max range.

        Args:
            cx, cy: Camera position in world coordinates (X=right, Y=forward)
            R_w: Camera-to-world rotation matrix (3x3). If None, assumes
                 camera points along +Y (forward in world frame).
        """
        # Use fov_x_deg for horizontal FOV (works in both physical and FOV modes)
        fov_deg = self.config.get("fov_x_deg", 70.0)
        max_range_m = self.config.get("max_range_mm", 4000) / 1000.0

        # Get camera forward direction in world X-Y plane
        # Camera's forward vector in camera frame is [0, 0, 1] (Z axis)
        # Transform to world frame: forward_world = R_w @ [0, 0, 1]
        if R_w is not None:
            forward_world = R_w[:, 2]  # Third column is Z axis in world frame
            # Project onto X-Y plane (ignore Z component for heading)
            fx, fy = forward_world[0], forward_world[1]
            # Normalize
            norm = np.sqrt(fx**2 + fy**2)
            if norm > 0.001:
                fx, fy = fx / norm, fy / norm
            else:
                fx, fy = 0.0, 1.0  # Default: pointing forward
        else:
            fx, fy = 0.0, 1.0  # Default: pointing along +Y

        # Calculate cone edges at half FOV angle
        half_fov = np.radians(fov_deg / 2.0)
        cos_hf = np.cos(half_fov)
        sin_hf = np.sin(half_fov)

        # Rotate forward vector by ±half_fov to get cone edges
        # For left edge: rotate forward by +half_fov
        left_fx = fx * cos_hf - fy * sin_hf
        left_fy = fx * sin_hf + fy * cos_hf

        # For right edge: rotate forward by -half_fov
        right_fx = fx * cos_hf + fy * sin_hf
        right_fy = -fx * sin_hf + fy * cos_hf

        # Calculate cone tip points (max range)
        left_tip_x = cx + left_fx * max_range_m
        left_tip_y = cy + left_fy * max_range_m

        right_tip_x = cx + right_fx * max_range_m
        right_tip_y = cy + right_fy * max_range_m

        # Define polygon vertices: camera position, left tip, right tip
        # Closing back to camera position
        xs = [cx, left_tip_x, right_tip_x, cx]
        ys = [cy, left_tip_y, right_tip_y, cy]

        self.map_camera_fov.setData(xs, ys)

        # Draw dead zone cone (red, from camera to min_depth)
        min_depth_m = self.config.get("min_depth_mm", 100) / 1000.0
        dz_left_tip_x = cx + left_fx * min_depth_m
        dz_left_tip_y = cy + left_fy * min_depth_m
        dz_right_tip_x = cx + right_fx * min_depth_m
        dz_right_tip_y = cy + right_fy * min_depth_m

        dz_xs = [cx, dz_left_tip_x, dz_right_tip_x, cx]
        dz_ys = [cy, dz_left_tip_y, dz_right_tip_y, cy]
        self.map_dead_zone.setData(dz_xs, dz_ys)

    def _update_map(
        self, result, depth, conf, conf_threshold, is_fresh=True, freeze_vo=False
    ):
        """Update the top-down map from the current frame's VO result.

        is_fresh:  True when this frame's VO result was just computed (not replayed
                   from cache). Only fresh results add points to the depth cloud,
                   preventing duplicates when scrubbing already-processed frames.
        freeze_vo: True when VO confidence is below threshold and the 'freeze VO'
                   mode is active. Prevents new VO positions from being added to
                   the path, keeping the last known valid position.
        """
        # Handle VO paths only if we have a result
        if result is not None:
            pos_x = result.get("pos_x", 0.0)
            pos_y = result.get("pos_y", 0.0)
            # Z is up per IMU convention, so the floor plane is X (right) vs Y (forward)

            # VO path — only advance when not frozen
            if not freeze_vo:
                self.map_vo_positions.append((pos_x, pos_y))
            if (
                self.map_show_vo_cb.isChecked()
                and len(self.map_vo_positions) > 1
                and not freeze_vo
            ):
                xs = [p[0] for p in self.map_vo_positions]
                ys = [p[1] for p in self.map_vo_positions]
                self.map_vo_curve.setData(xs, ys)

        # Fused path
        if self.ekf_enabled_cb.isChecked():
            fx, fy, fz = self.engine.ekf.get_position()
            self.map_fused_positions.append((fx, fy))
            if self.map_show_fused_cb.isChecked() and len(self.map_fused_positions) > 1:
                fxs = [p[0] for p in self.map_fused_positions]
                fys = [p[1] for p in self.map_fused_positions]
                self.map_fused_curve.setData(fxs, fys)

        # Current position marker (show fused if available, else VO)
        if self.ekf_enabled_cb.isChecked() and self.map_fused_positions:
            cx, cy = self.map_fused_positions[-1]
            R_w = self.engine.ekf.get_rotation_matrix()
        elif result is not None:
            cx, cy = result.get("pos_x", 0.0), result.get("pos_y", 0.0)
            R_w = result.get("R_world")
        else:
            cx, cy = 0.0, 0.0
            R_w = None
        self.map_current_marker.setData([cx], [cy])
        self._update_camera_fov(cx, cy, R_w)

        # Depth point cloud projection
        if not self.map_show_pts_cb.isChecked():
            return

        # Determine which mode: dense (all pixels) or sparse (tracked features)
        use_dense = self.config.get("dense_point_cloud", False)

        # Dense mode: project every Nth pixel (works even without VO)
        # Sparse mode: only project tracked feature points (requires VO result)
        if use_dense:
            # Dense mode works on every frame, not just fresh VO frames
            self._project_dense_points(depth, conf, conf_threshold)
        elif is_fresh and result is not None:
            # Sparse mode: only when VO produced a fresh result
            self._project_tracked_points(result, depth, conf, conf_threshold)

    def _get_projection_pose(self):
        """Get the pose (rotation matrix and position) to use for point projection.

        Returns:
            (R_w, pos): rotation matrix (3x3) and position (3,) in world coordinates
        """
        use_ekf = self.config.get("use_ekf_pose_for_points", True)

        if use_ekf and self.ekf_enabled_cb.isChecked():
            # Use fused EKF pose (smoother, less drift)
            R_w = self.engine.ekf.get_rotation_matrix()
            pos = np.array(self.engine.ekf.get_position())
        else:
            # Fallback to VO pose from current frame's result
            # This is set in _project_tracked_points or _project_dense_points
            return None, None

        return R_w, pos

    def _project_dense_points(self, depth, conf, conf_threshold):
        """Project all depth pixels to world coordinates (dense point cloud).

        Uses configurable pixel skip for downsampling and applies constraints.
        """
        R_w, pos = self._get_projection_pose()
        if R_w is None:
            # EKF not available and no VO result for this frame
            return

        skip = self.config.get("point_cloud_pixel_skip", 4)
        max_range_mm = 4000 if self.config.get("constrain_max_range", True) else 65535
        constrain_front = self.config.get("constrain_to_front", True)

        f = self.engine.vo.focal_length
        cx_cam, cy_cam = self.engine.vo.center

        new_pts = []
        h, w = depth.shape

        # Iterate over image with stride=skip for downsampling
        for v in range(0, h, skip):
            for u in range(0, w, skip):
                d_mm = float(depth[v, u])

                # Constraint: valid depth within sensor range
                if d_mm <= 0 or d_mm > max_range_mm:
                    continue

                # Constraint: confidence threshold
                if conf[v, u] < conf_threshold:
                    continue

                # Unproject to camera coordinates
                d_m = d_mm / 1000.0
                X_cam = (u - cx_cam) / f * d_m
                Y_cam = (v - cy_cam) / f * d_m
                Z_cam = d_m

                # Constraint: must be in front of camera
                if constrain_front and Z_cam <= 0.1:
                    continue

                # Transform to world coordinates
                P_world = R_w @ np.array([X_cam, Y_cam, Z_cam]) + pos
                new_pts.append((float(P_world[0]), float(P_world[1])))

        # Add to point cloud with size limit
        self._add_points_to_cloud(new_pts)

    def _project_tracked_points(self, result, depth, conf, conf_threshold):
        """Project only tracked feature points (sparse point cloud).

        Original behavior, but now uses EKF pose when configured.
        """
        use_ekf = self.config.get("use_ekf_pose_for_points", True)

        if use_ekf and self.ekf_enabled_cb.isChecked():
            # Use fused EKF pose
            R_w = self.engine.ekf.get_rotation_matrix()
            pos = np.array(self.engine.ekf.get_position())
        else:
            # Use VO pose from result
            R_w = result.get("R_world")
            if R_w is None:
                return
            pos = np.array(
                [
                    result.get("pos_x", 0.0),
                    result.get("pos_y", 0.0),
                    result.get("pos_z", 0.0),
                ]
            )

        max_range_mm = 4000 if self.config.get("constrain_max_range", True) else 65535
        constrain_front = self.config.get("constrain_to_front", True)

        f = self.engine.vo.focal_length
        cx_cam, cy_cam = self.engine.vo.center

        tracked_pts = result.get("tracked_pts", [])
        new_pts = []

        for pt in tracked_pts:
            u, v = pt[0], pt[1]
            iu, iv = int(round(u)), int(round(v))
            if not (0 <= iu < depth.shape[1] and 0 <= iv < depth.shape[0]):
                continue
            if conf[iv, iu] < conf_threshold:
                continue

            d_mm = float(depth[iv, iu])
            # Constraint: valid depth within sensor range
            if d_mm <= 0 or d_mm > max_range_mm:
                continue

            d_m = d_mm / 1000.0
            X_cam = (u - cx_cam) / f * d_m
            Y_cam = (v - cy_cam) / f * d_m
            Z_cam = d_m

            # Constraint: must be in front of camera
            if constrain_front and Z_cam <= 0.1:
                continue

            P_world = R_w @ np.array([X_cam, Y_cam, Z_cam]) + pos
            new_pts.append((float(P_world[0]), float(P_world[1])))

        self._add_points_to_cloud(new_pts)

    def _add_points_to_cloud(self, new_pts):
        """Add new points to the point cloud with size limit."""
        self.map_depth_points.extend(new_pts)
        max_pts = self.map_max_pts_spin.value()
        if len(self.map_depth_points) > max_pts:
            self.map_depth_points = self.map_depth_points[-max_pts:]

        if self.map_depth_points:
            dxs = [p[0] for p in self.map_depth_points]
            dys = [p[1] for p in self.map_depth_points]
            self.map_depth_scatter.setData(dxs, dys)

    def _on_range_changed(self):
        min_idx, max_idx = self.range_slider.value()
        if self.current_frame < min_idx:
            self.current_frame = min_idx
            self.slider.setValue(self.current_frame)
        elif self.current_frame > max_idx:
            self.current_frame = max_idx
            self.slider.setValue(self.current_frame)
        # Persist range so it's restored on next launch
        self.config["range_min"] = min_idx
        self.config["range_max"] = max_idx
        self._save_config()
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
        # Confidence threshold affects VO processing — must recompute
        self.engine.update_config(self.config)
        self._recompute_vo()

    def _on_speed_changed(self, value):
        self.config["playback_speed"] = value
        self._save_config()
        if self.playing:
            self.playback_timer.start(int(1000 / value))

    def _on_bilateral_toggled(self, checked):
        self.config["bilateral_enabled"] = checked
        self._save_config()
        self._update_bilateral_params()
        # Bilateral filter now applied to VO data — must recompute
        self.engine.update_config(self.config)
        self._recompute_vo()

    def _on_median_toggled(self, checked):
        self.config["median_enabled"] = checked
        self._save_config()
        self._update_median_params()
        # Median filter now applied to VO data — must recompute
        self.engine.update_config(self.config)
        self._recompute_vo()

    def _on_filter_changed(self):
        self.config["bilateral_d"] = self.bilateral_d.value()
        self.config["bilateral_sigma_color"] = self.bilateral_sigma_color.value()
        self.config["bilateral_sigma_space"] = self.bilateral_sigma_space.value()
        self.config["median_k"] = self.median_k.value()
        self._save_config()
        # Filter params affect VO data — must recompute
        self.engine.update_config(self.config)
        self._recompute_vo()

    def _update_bilateral_params(self):
        enabled = self.bilateral_cb.isChecked()
        for row in self.bilateral_rows:
            self.bilateral_layout.setRowVisible(row, enabled)

    def _update_median_params(self):
        enabled = self.median_cb.isChecked()
        self.median_layout.setRowVisible(1, enabled)

    def _on_vo_setting_changed(self):
        self.config["detector_threshold"] = self.vo_detector_spin.value()
        self.config["rejection_threshold"] = self.vo_rejection_spin.value()
        self.config["stationary_threshold"] = self.vo_stationary_spin.value()
        self._save_config()
        self.engine.update_config(self.config)
        self._recompute_vo()

    def _on_clahe_toggled(self, checked):
        self.config["clahe_enabled"] = checked
        self._save_config()
        self._update_clahe_params()
        self.engine.update_config(self.config)
        self._recompute_vo()
        for row in self.clahe_rows:
            self.clahe_layout.setRowVisible(row, checked)

    def _on_clahe_changed(self):
        self.config["clahe_clip_limit"] = self.clahe_clip_spin.value() / 10.0
        self.config["clahe_grid_size"] = self.clahe_grid_spin.value()
        self._save_config()
        self.engine.update_config(self.config)
        self._recompute_vo()

    def _on_camera_params_changed(self):
        """Handle changes to camera focal length and sensor size."""
        self.config["focal_length_mm"] = self.focal_spin.value()
        self.config["sensor_size"] = self.sensor_combo.currentText()
        self._save_config()
        # Update VO camera setup
        self.engine.vo._setup_camera()
        # Update computed FOV display and sync fov_x_deg
        self._update_computed_fov()
        # Sync fov_x_deg from computed value for cone visualization
        try:
            f_mm = self.focal_spin.value()
            sensor_size = self.sensor_combo.currentText()
            sizes = {
                "1/6": 3.0,
                "1/4": 4.0,
                "1/3": 6.0,
                "1/2.3": 7.0,
                "1/2": 8.0,
                "2/3": 11.0,
                "1": 16.0,
            }
            diag_mm = sizes.get(sensor_size, 3.0)
            aspect = 4 / 3
            sensor_width = diag_mm / np.sqrt(1 + aspect**2)
            fov_x = 2 * np.degrees(np.arctan(sensor_width / (2 * f_mm)))
            self.config["fov_x_deg"] = fov_x
            self._save_config()
        except:
            pass
        # Must recompute VO since projection changes
        self._recompute_vo()

    def _on_max_range_changed(self, value):
        """Handle changes to max range (mm)."""
        self.config["max_range_mm"] = value
        self._save_config()
        # Clear point cloud since max range affects filtering
        self.map_depth_points.clear()
        self.update_display()

    def _on_min_depth_changed(self, value):
        """Handle changes to min depth/dead zone (mm)."""
        self.config["min_depth_mm"] = value
        self._save_config()
        # Clear point cloud since min depth affects filtering
        self.map_depth_points.clear()
        self.update_display()

    def _on_camera_mode_toggled(self, checked):
        """Handle toggle between physical and FOV camera modes."""
        self.config["use_fov_mode"] = checked
        self._save_config()
        self._update_camera_mode_visibility()
        # Update camera setup and recompute
        self.engine.vo._setup_camera()
        self._recompute_vo()

    def _on_fov_params_changed(self):
        """Handle changes to FOV mode parameters."""
        self.config["fov_x_deg"] = self.fov_x_spin.value()
        self.config["fov_y_deg"] = self.fov_y_spin.value()
        self.config["resolution_x"] = self.res_x_spin.value()
        self.config["resolution_y"] = self.res_y_spin.value()
        self._save_config()
        # Update VO camera setup
        self.engine.vo._setup_camera()
        # Update computed focal length display
        self._update_computed_focal_length()
        # Must recompute VO since projection changes
        self._recompute_vo()

    def _update_camera_mode_visibility(self):
        """Update UI visibility based on camera mode."""
        use_fov = self.use_fov_mode_cb.isChecked()
        self.physical_mode_widget.setVisible(not use_fov)
        self.fov_mode_widget.setVisible(use_fov)
        # Update computed values display
        if use_fov:
            self._update_computed_focal_length()
        else:
            self._update_computed_fov()

    def _update_computed_fov(self):
        """Update the computed FOV labels in physical mode."""
        try:
            f_mm = self.focal_spin.value()
            sensor_size = self.sensor_combo.currentText()
            # Compute sensor dimensions
            sizes = {
                "1/6": 3.0,
                "1/4": 4.0,
                "1/3": 6.0,
                "1/2.3": 7.0,
                "1/2": 8.0,
                "2/3": 11.0,
                "1": 16.0,
            }
            diag_mm = sizes.get(sensor_size, 3.0)
            aspect = 4 / 3
            sensor_width = diag_mm / np.sqrt(1 + aspect**2)
            sensor_height = sensor_width * aspect
            # Compute FOV
            fov_x = 2 * np.degrees(np.arctan(sensor_width / (2 * f_mm)))
            fov_y = 2 * np.degrees(np.arctan(sensor_height / (2 * f_mm)))
            self.computed_fov_x_label.setText(f"{fov_x:.1f}°")
            self.computed_fov_y_label.setText(f"{fov_y:.1f}°")
        except:
            self.computed_fov_x_label.setText("--")
            self.computed_fov_y_label.setText("--")

    def _update_computed_focal_length(self):
        """Update the computed focal length label in FOV mode."""
        try:
            fov_x = np.radians(self.fov_x_spin.value())
            res_x = self.res_x_spin.value()
            focal_px = (res_x / 2) / np.tan(fov_x / 2)
            # Compute focal length in mm assuming 1/6" sensor (3mm diagonal)
            # This is an approximation since we don't have physical sensor size in FOV mode
            sensor_width_mm = 2.4  # Approximate for 1/6" sensor with 4:3 aspect
            focal_mm = focal_px * (sensor_width_mm / res_x)
            self.computed_focal_label.setText(f"{focal_px:.1f} px ({focal_mm:.1f} mm)")
        except:
            self.computed_focal_label.setText("--")

    def _update_clahe_params(self):
        enabled = self.clahe_cb.isChecked()
        self.clahe_clip_spin.setEnabled(enabled)
        self.clahe_grid_spin.setEnabled(enabled)

    def session_selected(self, name):
        if not name:
            return
        self.playback_timer.stop()
        self.play_btn.setText("Play")
        self.playing = False
        data_dir = Path(__file__).parent / "data"
        self.session_dir = data_dir / name

        frames_dir = self.session_dir / "frames"
        if not frames_dir.exists():
            return

        num_frames = self.frame_loader.set_session(frames_dir)
        if num_frames == 0:
            return

        # We need a reference to at least one frame to get timestamps if we don't have IMU
        # but let's assume we can at least get metadata from the loader
        first_frame = self.frame_loader.get_frame(0)
        if first_frame is None:
            return

        # Get all timestamps for IMU interpolation (this is still a bit heavy but much less than full frames)
        # Actually, let's just read the timestamps from the filenames or metadata if possible
        # For now, let's just load the timestamps only.
        frame_files = sorted(frames_dir.glob("frame_*.npz"))
        frame_ts = []
        for f in frame_files:
            # We can use np.load with mmap_mode to just read the timestamp without loading the whole depth/conf
            with np.load(f, mmap_mode="r") as data:
                frame_ts.append(data["timestamp_ns"])

        imu_path = self.session_dir / "imu_data.csv"
        if imu_path.exists():
            self.imu_data = np.genfromtxt(imu_path, delimiter=",", skip_header=1)
            imu_ts = self.imu_data[:, 0]

            for key in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
                idx = [
                    "timestamp_ns",
                    "acc_x",
                    "acc_y",
                    "acc_z",
                    "gyro_x",
                    "gyro_y",
                    "gyro_z",
                ].index(key)
                self.imu_interpolated[key] = np.interp(
                    frame_ts, imu_ts, self.imu_data[:, idx]
                ).tolist()
        else:
            self.imu_data = None
            self.imu_interpolated = {
                key: [0.0] * num_frames
                for key in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
            }

        self.engine.reset(self.imu_data)

        if self.log_ekf_cb.isChecked():
            self.engine.start_logging(self.session_dir / "ekf_debug.log")
        else:
            self.engine.stop_logging()

        # Reset map and denoised state
        self._clear_map()

        max_frame = num_frames - 1
        self.slider.setMaximum(max_frame)
        self.range_slider.setRange(0, max_frame)

        # Restore saved range and frame only if this is the same session as last time
        if self.config.get("last_session") == name:
            saved_min = int(self.config.get("range_min", 0))
            saved_max = min(int(self.config.get("range_max", max_frame)), max_frame)
            saved_frame = min(int(self.config.get("last_frame", 0)), saved_max)
        else:
            saved_min, saved_max, saved_frame = 0, max_frame, 0

        # Save which session is active (after reading saved values above)
        self.config["last_session"] = name
        self._save_config()

        self.range_slider.setValue(saved_min, saved_max)
        self.current_frame = saved_frame
        self.last_displayed_frame = -1
        self.slider.setValue(saved_frame)

        self.view_combo.setCurrentText(self.config.get("view_mode", "depth"))

        self.vo_results = [None] * num_frames
        filt_keys = {
            "acc_x_filt",
            "acc_y_filt",
            "acc_z_filt",
            "gyro_x_filt",
            "gyro_y_filt",
            "gyro_z_filt",
        }
        for key in self.graph_data:
            if key in self.imu_interpolated:
                self.graph_data[key] = list(self.imu_interpolated[key])
            elif key in filt_keys:
                self.graph_data[key] = [0.0] * num_frames
            elif key in ["pos_fused_x", "pos_fused_y", "pos_fused_z"]:
                self.graph_data[key] = [0.0] * num_frames
            else:
                self.graph_data[key] = [0] * num_frames

        # Must run AFTER graph_data is initialized so filtered series aren't overwritten
        self._recompute_imu_filter()

        self.update_display()

    def _recompute_vo(self):
        num_frames = len(self.frame_loader.frame_paths)
        if num_frames == 0:
            return
        self.engine.reset(self.imu_data)
        self._recompute_imu_filter()
        self._clear_map()
        self.last_displayed_frame = -1
        self.vo_results = [None] * num_frames
        for key in [
            "tracked",
            "rejected",
            "ratio",
            "pos_x",
            "pos_y",
            "pos_z",
            "mean_depth",
            "mean_conf",
            "fps",
            "vo_score",
        ]:
            self.graph_data[key] = [0] * num_frames
        for key in ["pos_fused_x", "pos_fused_y", "pos_fused_z"]:
            self.graph_data[key] = [0.0] * num_frames
        self.update_display()

    def slider_moved(self, value):
        self.current_frame = value
        # Persist frame position so it's restored on next launch
        self.config["last_frame"] = value
        self._save_config()
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

    def prev_frame(self):
        min_idx, max_idx = self.range_slider.value()
        if self.current_frame > min_idx:
            self.current_frame -= 1
            self.slider.setValue(self.current_frame)
        else:
            self.current_frame = max_idx
            self.slider.setValue(self.current_frame)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.toggle_play()
        elif key == Qt.Key.Key_Right:
            if self.playing:
                self.toggle_play()  # stop playback before manual step
            self.next_frame()
        elif key == Qt.Key.Key_Left:
            if self.playing:
                self.toggle_play()
            self.prev_frame()
        else:
            super().keyPressEvent(event)

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
        if obj == self.image_label:
            if event.type() == QEvent.Type.MouseMove:
                if self.current_depth is not None and self.current_conf is not None:
                    pixmap = self.image_label.pixmap()
                    if pixmap:
                        # Map from displayed (scaled) pixel coords to original array coords
                        label_w = self.image_label.width()
                        label_h = self.image_label.height()
                        px_w = pixmap.width()
                        px_h = pixmap.height()
                        # Pixmap is centered inside label — find offset
                        offset_x = (label_w - px_w) / 2
                        offset_y = (label_h - px_h) / 2
                        img_x = event.position().x() - offset_x
                        img_y = event.position().y() - offset_y
                        orig_w = self.current_depth.shape[1]
                        orig_h = self.current_depth.shape[0]
                        x = int(img_x * orig_w / px_w)
                        y = int(img_y * orig_h / px_h)
                        if 0 <= x < orig_w and 0 <= y < orig_h:
                            depth_val = int(self.current_depth[y, x])
                            conf_val = self.current_conf[y, x]
                            conf_thresh = self.config.get("conf_threshold", 10)
                            valid = "✓" if conf_val >= conf_thresh else "✗"
                            self.pixel_info_label.setText(
                                f"x: {x:3d}  y: {y:3d}  "
                                f"depth: {depth_val:5d} mm  "
                                f"conf: {conf_val:.1f}  {valid}"
                            )
                        else:
                            self.pixel_info_label.setText(
                                "Hover over image for pixel info"
                            )
            elif event.type() == QEvent.Type.Leave:
                self.pixel_info_label.setText("Hover over image for pixel info")
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
        data_len = len(self.graph_data.get("tracked", []))
        if min_idx <= x <= max_idx and x < data_len and self.graph_data.get("tracked"):
            lines = [f"<div style='text-align: center'><b>Frame {x}</b><br>"]
            for key, curve in self.plot_curves.items():
                if curve.isVisible() and x < len(self.graph_data.get(key, [])):
                    value = self.graph_data[key][x]
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
        num_frames = len(self.frame_loader.frame_paths)
        if num_frames == 0 or not self.graph_data["tracked"]:
            return

        min_idx, max_idx = self.range_slider.value()
        max_idx = min(max_idx, num_frames - 1)
        x = list(range(min_idx, max_idx + 1))

        for key, curve in self.plot_curves.items():
            if not curve.isVisible():
                continue
            data = self.graph_data.get(key)
            if data is None or len(data) == 0:
                continue
            data_len = len(data)
            slice_end = min(max_idx + 1, data_len)
            data_slice = data[min_idx:slice_end]
            if not data_slice:
                continue
            x_slice = list(range(min_idx, min_idx + len(data_slice)))
            data_arr = np.array(data_slice)
            max_val = (
                np.abs(data_arr).max()
                if data_arr.size > 0 and np.abs(data_arr).max() != 0
                else 1
            )
            normalized = data_arr / max_val * 100
            curve.setData(x_slice, normalized)

        self.playhead_line.setValue(self.current_frame)

    def update_display(self):
        num_frames = len(self.frame_loader.frame_paths)
        if num_frames == 0:
            return

        # Detect non-sequential frame access (scrubbing) and reset stateful algorithms
        jumped = (
            self.last_displayed_frame >= 0
            and self.current_frame != self.last_displayed_frame + 1
        )
        if jumped:
            self.engine.reset(self.imu_data)
            self._clear_paths()

        self.last_displayed_frame = self.current_frame

        frame = self.frame_loader.get_frame(self.current_frame)
        if frame is None:
            return

        view_mode = self.view_combo.currentText()
        conf_threshold = self.config.get("conf_threshold", 10)

        depth = frame["depth"]
        conf = frame["conf"]
        ts = frame["timestamp_ns"]

        was_cached = self.vo_results[self.current_frame] is not None

        # Prepare IMU data for engine
        gyro = np.array(
            [
                self.imu_interpolated["gyro_x"][self.current_frame],
                self.imu_interpolated["gyro_y"][self.current_frame],
                self.imu_interpolated["gyro_z"][self.current_frame],
            ]
        )
        accel = np.array(
            [
                self.imu_interpolated["acc_x"][self.current_frame],
                self.imu_interpolated["acc_y"][self.current_frame],
                self.imu_interpolated["acc_z"][self.current_frame],
            ]
        )

        if self.imu_filter_enabled:
            gyro = np.array(
                [
                    self.imu_filtered[k][self.current_frame]
                    for k in ["gyro_x", "gyro_y", "gyro_z"]
                ]
            )
            accel = np.array(
                [
                    self.imu_filtered[k][self.current_frame]
                    for k in ["acc_x", "acc_y", "acc_z"]
                ]
            )

        prev_ts = None
        if self.current_frame > 0:
            prev_frame = self.frame_loader.get_frame(self.current_frame - 1)
            if prev_frame is not None:
                prev_ts = prev_frame["timestamp_ns"]
        dt = (ts - prev_ts) / 1e9 if prev_ts else 0.033

        if self.vo_enabled_cb.isChecked() and not was_cached:
            vo_res, fused_pos, denoised = self.engine.process_frame(
                depth,
                conf,
                ts,
                gyro,
                accel,
                dt,
                use_denoised=self.vo_use_denoised_cb.isChecked(),
                denoise_alpha=self.denoise_alpha,
            )
            self.vo_results[self.current_frame] = vo_res
            self.current_depth = denoised if view_mode == "denoised" else depth
            self.current_conf = conf

            for key in [
                "tracked",
                "rejected",
                "ratio",
                "pos_x",
                "pos_y",
                "pos_z",
                "mean_depth",
                "mean_conf",
                "fps",
                "vo_score",
            ]:
                self.graph_data[key][self.current_frame] = vo_res.get(key, 0)

            self.graph_data["pos_fused_x"][self.current_frame] = fused_pos[0]
            self.graph_data["pos_fused_y"][self.current_frame] = fused_pos[1]
            self.graph_data["pos_fused_z"][self.current_frame] = fused_pos[2]

            result = vo_res
            display_depth = self.current_depth
        else:
            result = self.vo_results[self.current_frame]
            self.current_conf = conf
            if result and self.ekf_enabled_cb.isChecked():
                vo_pos = [result["pos_x"], result["pos_y"], result["pos_z"]]
                fused_pos = self.engine.update_ekf(
                    gyro, accel, dt, vo_pos, result.get("vo_score", 0.0)
                )
                self.graph_data["pos_fused_x"][self.current_frame] = fused_pos[0]
                self.graph_data["pos_fused_y"][self.current_frame] = fused_pos[1]
                self.graph_data["pos_fused_z"][self.current_frame] = fused_pos[2]

            # If we are in denoised mode but result was cached, we ideally want to show denoised depth.
            # However, denoising is temporal and stateful. Since we are in 'else', we are either:
            # 1. Sequential playback but VO was cached (this happens if user processed range before).
            # 2. Scrubbing (was_cached will be False because we reset vo_results? No, vo_results persists).
            # If it's sequential, we can still compute denoised depth.
            if view_mode == "denoised" and self.ekf_enabled_cb.isChecked():
                # Re-run denoising only to keep state consistent for display
                denoised = self.engine.compute_denoised(
                    depth, conf, conf_threshold, self.denoise_alpha
                )
                self.current_depth = denoised
                display_depth = denoised
            else:
                self.current_depth = depth
                display_depth = depth

        masked_depth = np.where(conf >= conf_threshold, depth, 0)
        invalid_mask = conf < conf_threshold

        # Update top-down map with latest VO result
        vo_score = result.get("vo_score", 0.0) if result else 0.0
        min_vo_conf = self.min_vo_conf_spin.value()
        conf_ok = (min_vo_conf == 0.0) or (vo_score >= min_vo_conf)
        low_conf_mode = self.config.get("low_conf_mode", "both")
        freeze_vo_path = not conf_ok and low_conf_mode in ("freeze_vo", "both")

        self._update_map(
            result,
            depth,
            conf,
            conf_threshold,
            is_fresh=not was_cached,
            freeze_vo=freeze_vo_path,
        )

        # --- Visualization ---
        def _depth_to_rgb(d_arr):
            """Normalize a depth array to uint8 RGB via Viridis colormap."""
            d_filtered = d_arr.copy()
            if self.config.get("bilateral_enabled", False):
                bd = self.config.get("bilateral_d", 9)
                bsc = self.config.get("bilateral_sigma_color", 200)
                bss = self.config.get("bilateral_sigma_space", 20)
                d_filtered = cv2.bilateralFilter(
                    d_filtered.astype(np.float32), bd, bsc, bss
                )
            if self.config.get("median_enabled", False):
                mk = self.config.get("median_k", 5)
                d_filtered = cv2.medianBlur(d_filtered.astype(np.float32), mk)
            dmax = float(d_filtered.max())
            norm = (
                np.clip(d_filtered * 255 / dmax, 0, 255).astype(np.uint8)
                if dmax > 0
                else d_filtered.astype(np.uint8)
            )
            rgb = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
            rgb[invalid_mask] = [40, 40, 40]
            return rgb

        if view_mode == "depth":
            data = _depth_to_rgb(masked_depth)
        elif view_mode == "conf":
            conf_max = float(conf.max()) if conf.max() > 0 else 1.0
            norm = (conf * 255 / conf_max).astype(np.uint8)
            data = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
        elif view_mode == "denoised":
            if self.ekf_enabled_cb.isChecked():
                data = _depth_to_rgb(display_depth)
            else:
                # EKF off: show regular depth with a notice overlay
                data = _depth_to_rgb(masked_depth)
                cv2.putText(
                    data,
                    "Enable EKF for denoised view",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 200, 0),
                    1,
                )
        else:
            data = _depth_to_rgb(masked_depth)

        if self.show_tracked_cb.isChecked() or self.show_rejected_cb.isChecked():
            if self.current_frame < len(self.vo_results):
                overlay_result = self.vo_results[self.current_frame]

                if overlay_result is not None:
                    if self.show_tracked_cb.isChecked():
                        for pt in overlay_result.get("tracked_pts", []):
                            x, y = int(pt[0]), int(pt[1])
                            if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
                                cv2.circle(data, (x, y), 3, (0, 255, 0), -1)

                    if self.show_rejected_cb.isChecked():
                        for pt in overlay_result.get("rejected_pts", []):
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

        for key, label in self.series_labels.items():
            if self.current_frame < len(self.graph_data.get(key, [])):
                value = self.graph_data[key][self.current_frame]
                # Show 2 decimal places for position values, 1 for others
                if "pos" in key:
                    label.setText(f"{value:.2f}")
                else:
                    label.setText(f"{value:.1f}")
            else:
                label.setText("0.0")

        self._update_graph()

    def _on_process_range_clicked(self):
        if not self.session_dir:
            return

        min_idx, max_idx = self.range_slider.value()
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(min_idx, max_idx)
        self.progress_bar.setValue(min_idx)

        # Use a fresh engine instance for the worker to avoid thread safety issues
        worker_engine = ProcessingEngine(self.config)
        worker_engine.reset(self.imu_data)
        if self.log_ekf_cb.isChecked():
            worker_engine.start_logging(self.session_dir / "ekf_debug_worker.log")

        self.process_thread = QThread()
        self.process_worker = ProcessingWorker(
            worker_engine, self.frame_loader, self.imu_interpolated, min_idx, max_idx
        )
        self.process_worker.moveToThread(self.process_thread)

        self.process_thread.started.connect(self.process_worker.run)
        self.process_worker.progress.connect(self.progress_bar.setValue)
        self.process_worker.finished.connect(self._on_processing_finished)
        self.process_worker.finished.connect(self.process_thread.quit)
        self.process_worker.finished.connect(self.process_worker.deleteLater)
        self.process_thread.finished.connect(self.process_thread.deleteLater)

        self.process_thread.start()

    def _on_processing_finished(self, results):
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        min_idx, _ = self.range_slider.value()
        for i, res in enumerate(results):
            frame_idx = min_idx + i
            if res is not None:
                vo_res, fused_pos = res
                self.vo_results[frame_idx] = vo_res
                if vo_res:
                    for key in [
                        "tracked",
                        "rejected",
                        "ratio",
                        "pos_x",
                        "pos_y",
                        "pos_z",
                        "mean_depth",
                        "mean_conf",
                        "fps",
                        "vo_score",
                    ]:
                        self.graph_data[key][frame_idx] = vo_res.get(key, 0)

                self.graph_data["pos_fused_x"][frame_idx] = fused_pos[0]
                self.graph_data["pos_fused_y"][frame_idx] = fused_pos[1]
                self.graph_data["pos_fused_z"][frame_idx] = fused_pos[2]

        self._update_graph()
        self.update_display()


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    viewer = DataViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
