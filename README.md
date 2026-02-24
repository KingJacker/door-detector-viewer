# Door Detector Tuner - Data Viewer

A PySide6/PyQtGraph application for visualizing depth camera recordings with visual odometry, IMU fusion, and real-time parameter tuning. Built for developing a door detection and localization system for electric wheelchair navigation.

## Overview

This tool replays recorded Time-of-Flight depth camera sessions and provides:

- Depth/confidence image visualization with configurable filtering
- Visual odometry (FAST + optical flow + Essential Matrix)
- Extended Kalman Filter fusion of VO + IMU data
- Real-time graph plotting of 18 data series
- Full parameter tuning UI with live feedback

## Requirements

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
uv sync
```

## Usage

```bash
uv run viewer.py
```

Place recording sessions in the `data/` directory and select them from the session dropdown.

## Data Format

Each recording session is a timestamped directory:

```
data/<YYYYMMDD_HHMMSS>/
├── metadata.json         # Session info (session_id, total_frames, end_time)
├── frames_index.csv      # Frame index with timestamps
├── imu_data.csv          # 6-axis IMU: timestamp_ns, acc_xyz, gyro_xyz
├── performance.csv       # Per-frame timing data
└── frames/
    ├── frame_000000.npz  # depth, conf, amplitude, timestamp_ns
    ├── frame_000001.npz
    └── ...
```

**Frame arrays** (240x180, float32):

| Key | Description |
|-----|-------------|
| `depth` | Depth map in mm |
| `conf` | Per-pixel confidence (0-545) |
| `amplitude` | IR amplitude image |
| `timestamp_ns` | Nanosecond timestamp |

**IMU CSV columns**: `timestamp_ns, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`

IMU data is sampled at a higher rate than frames and interpolated to frame timestamps at load time.

## Architecture

```
viewer.py        Main application (DataViewer QMainWindow)
vo.py            Visual odometry engine
ekf.py           Extended Kalman Filter for VO+IMU fusion
range_slider.py  Custom dual-handle range slider widget
config.json      Persistent UI settings
```

### Layout

Three-panel splitter:

- **Left sidebar**: Camera info, graph series toggles with live values
- **Center**: Image display, frame slider, time-series graph, playback controls
- **Right sidebar** (scrollable): All tunable parameters

## Features

### Visual Odometry (`vo.py`)

Depth-based visual odometry pipeline:

1. FAST feature detection on preprocessed depth image
2. Lucas-Kanade pyramidal optical flow tracking
3. Quality filtering by optical flow error threshold
4. Essential Matrix estimation via RANSAC
5. Pose recovery and translation scaling using median depth
6. Global position accumulation

Configurable parameters: detector threshold, rejection threshold, stationary threshold.

### Extended Kalman Filter (`ekf.py`)

10-state EKF fusing IMU predictions with VO measurements:

- **State vector**: position (3), velocity (3), quaternion orientation (4)
- **Predict**: Quaternion integration from gyroscope, gravity-compensated acceleration for position/velocity
- **Update**: VO position as measurement, confidence-weighted measurement noise
- **Gravity**: Auto-detected from first ~100 IMU samples
- **Fixed height constraint**: Soft constraint blending estimated Z toward a user-specified height, with tunable strength (for fixed-mount scenarios)

### IMU Low-Pass Filter

Exponential moving average on all 6 IMU channels before EKF input:

```
filtered = alpha * raw + (1 - alpha) * previous
```

### Image Filters

- **Bilateral filter**: Edge-preserving smoothing (configurable d, sigma_color, sigma_space)
- **Median filter**: Salt-and-pepper noise removal (configurable kernel size)
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization for depth preprocessing

### Graph

Real-time normalized (0-100%) time-series plot with 18 series:

| Category | Series |
|----------|--------|
| VO stats | tracked, rejected, ratio |
| VO position | pos_x, pos_y, pos_z |
| Fused position | pos_fused_x, pos_fused_y, pos_fused_z |
| Frame stats | mean_depth, mean_conf, fps |
| Accelerometer | acc_x, acc_y, acc_z |
| Gyroscope | gyro_x, gyro_y, gyro_z |

Features: playhead line, range slider for sub-range selection, interactive tooltip on hover, per-series color-coded toggles with live value display.

### UI Settings (Right Sidebar)

All settings use a `{label}: {slider} {spinbox}` format and persist to `config.json`.

Settings groups collapse when their enable checkbox is unchecked:

- **VO Settings**: tracking toggle, overlay toggles, detector/rejection/stationary thresholds
- **CLAHE**: clip limit, grid size
- **Bilateral filter**: d, sigma color, sigma space
- **Median filter**: kernel size
- **EKF Settings**: process noise, measurement noise, fixed height constraint (height + strength)
- **IMU Filter**: low-pass alpha

## Camera

Default configuration targets a ToF depth camera with:

- Focal length: 3.4mm
- Sensor size: 1/6"
- Resolution: 240x180 (QVGA)

## Project Context

This viewer is part of a door detection and localization system designed for electric wheelchairs. The system uses a depth camera mounted at a fixed height to detect doors and safely navigate through them. The EKF fusion with fixed-height constraint is designed for this fixed-mount scenario.
