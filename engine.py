import numpy as np
from vo import VisualOdometry
from ekf import ExtendedKalmanFilter


class ProcessingEngine:
    def __init__(self, config):
        self.config = config
        self.vo = VisualOdometry(config)
        self.ekf = ExtendedKalmanFilter()

        # Internal state
        self.denoised_depth = None
        self.prev_ekf_pos = None
        self.prev_ekf_R = None
        self.log_file = None

        # Axis mapping [x_idx, y_idx, z_idx], signs [x_s, y_s, z_s]
        # Default: identity mapping
        self.axis_mapping = [0, 1, 2]
        self.axis_signs = [1, 1, 1]

    def reset(self, imu_data=None):
        self.vo.reset()
        self.ekf.reset()
        self.denoised_depth = None
        self.prev_ekf_pos = None
        self.prev_ekf_R = None
        if imu_data is not None:
            self.ekf.detect_up_direction(imu_data)

    def update_config(self, config):
        self.config = config
        self.vo.update_config(config)
        # Update EKF params
        self.ekf.process_noise = config.get("ekf_process_noise", 0.1)
        self.ekf.measurement_noise = config.get("ekf_measure_noise", 1.0)
        self.ekf.fixed_height_enabled = config.get("fixed_height_enabled", False)
        self.ekf.fixed_height = config.get("fixed_height", 1.2)
        self.ekf.height_strength = config.get("height_strength", 0.5)
        self.ekf.max_speed = config.get("max_speed", 2.0)
        self.ekf.max_rotation_speed = config.get("max_rotation_speed", 1.5)
        self.ekf.lateral_damping = config.get("lateral_damping", 0.1)

        # Update mapping from config if present
        if "imu_axis_mapping" in config:
            self.axis_mapping = config["imu_axis_mapping"]
        if "imu_axis_signs" in config:
            self.axis_signs = config["imu_axis_signs"]

    def start_logging(self, path):
        if self.log_file:
            self.log_file.close()
        try:
            self.log_file = open(path, "w")
            self.log_file.write(
                "timestamp_ns,dt,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z,fused_x,fused_y,fused_z,q_w,q_x,q_y,q_z\n"
            )
        except Exception as e:
            print(f"Failed to start EKF logging: {e}")
            self.log_file = None

    def stop_logging(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def compute_denoised(self, depth, conf, conf_threshold, denoise_alpha):
        H, W = depth.shape
        f = self.vo.focal_length
        cx, cy = self.vo.center

        valid = (conf >= conf_threshold) & (depth > 0)
        cur_depth_f = depth.astype(np.float32)

        pos_cur = np.array(self.ekf.get_position())
        R_cur = self.ekf.get_rotation_matrix()

        if self.denoised_depth is None:
            self.denoised_depth = cur_depth_f.copy()
            self.prev_ekf_pos = pos_cur.copy()
            self.prev_ekf_R = R_cur.copy()
            return self.denoised_depth

        R_rel = self.prev_ekf_R.T @ R_cur
        t_rel = self.prev_ekf_R.T @ (pos_cur - self.prev_ekf_pos)

        v_idx, u_idx = np.where(valid)
        if len(u_idx) == 0:
            self.prev_ekf_pos = pos_cur.copy()
            self.prev_ekf_R = R_cur.copy()
            return self.denoised_depth

        d_m = cur_depth_f[v_idx, u_idx] / 1000.0
        X_cur = (u_idx - cx) / f * d_m
        Y_cur = (v_idx - cy) / f * d_m
        Z_cur = d_m
        pts_cur = np.stack([X_cur, Y_cur, Z_cur], axis=1)

        pts_prev = (R_rel @ pts_cur.T).T + t_rel
        valid_z = pts_prev[:, 2] > 0.01
        u_prev = np.where(valid_z, f * pts_prev[:, 0] / pts_prev[:, 2] + cx, -1).astype(
            np.int32
        )
        v_prev = np.where(valid_z, f * pts_prev[:, 1] / pts_prev[:, 2] + cy, -1).astype(
            np.int32
        )

        in_bounds = (
            valid_z & (u_prev >= 0) & (u_prev < W) & (v_prev >= 0) & (v_prev < H)
        )
        new_denoised = (
            self.denoised_depth.copy()
        )  # Actually maybe we should use the previous denoised as base?
        # The previous implementation used self.denoised_depth.copy()

        u_v = u_idx[in_bounds]
        v_v = v_idx[in_bounds]
        u_p = u_prev[in_bounds]
        v_p = v_prev[in_bounds]

        prev_vals = self.denoised_depth[v_p, u_p]
        cur_vals = cur_depth_f[v_v, u_v]

        prev_valid = prev_vals > 0
        new_denoised[v_v[prev_valid], u_v[prev_valid]] = (
            denoise_alpha * cur_vals[prev_valid]
            + (1 - denoise_alpha) * prev_vals[prev_valid]
        )
        new_denoised[v_v[~prev_valid], u_v[~prev_valid]] = cur_vals[~prev_valid]

        u_nb = u_idx[~in_bounds]
        v_nb = v_idx[~in_bounds]
        new_denoised[v_nb, u_nb] = cur_depth_f[v_nb, u_nb]
        # Checked viewer.py line 1294: new_denoised[v_nb, u_nb] = cur_depth_f[v_nb, u_nb]
        # I will fix it here.

        self.denoised_depth = new_denoised
        self.prev_ekf_pos = pos_cur.copy()
        self.prev_ekf_R = R_cur.copy()
        return self.denoised_depth

    def update_ekf(self, gyro_raw, accel_raw, dt, vo_pos=None, vo_score=0.0, timestamp_ns=0):
        # 1. IMU conversion and remapping
        gyro_rads = np.deg2rad(gyro_raw)
        accel_ms2 = accel_raw * 9.81

        # Apply axis mapping and signs
        m = self.axis_mapping
        s = self.axis_signs
        gyro_mapped = np.array([gyro_rads[m[0]] * s[0], gyro_rads[m[1]] * s[1], gyro_rads[m[2]] * s[2]])
        accel_mapped = np.array([accel_ms2[m[0]] * s[0], accel_ms2[m[1]] * s[1], accel_ms2[m[2]] * s[2]])

        # 2. EKF Predict
        self.ekf.predict(gyro_mapped, accel_mapped, dt)

        # 3. EKF Update if VO position provided
        if vo_pos is not None:
            min_conf = self.config.get("min_vo_confidence", 0.0)
            if vo_score >= min_conf:
                self.ekf.update(vo_pos, vo_score)

        fused_pos = self.ekf.get_position()
        euler = self.ekf.get_euler_angles()

        # 4. Logging
        if self.log_file:
            q = self.ekf.q
            g = gyro_mapped
            a = accel_mapped
            p = fused_pos
            self.log_file.write(
                f"{timestamp_ns},{dt:.6f},{g[0]:.6f},{g[1]:.6f},{g[2]:.6f},"
                f"{a[0]:.6f},{a[1]:.6f},{a[2]:.6f},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},"
                f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}\n"
            )

        return fused_pos, euler

    def process_frame(
        self,
        depth,
        conf,
        timestamp_ns,
        gyro_raw,
        accel_raw,
        dt,
        use_denoised=False,
        denoise_alpha=0.3,
    ):
        conf_threshold = self.config.get("conf_threshold", 10)

        # 1. Denoising if requested
        denoised = None
        if use_denoised or self.config.get("view_mode") == "denoised":
            denoised = self.compute_denoised(depth, conf, conf_threshold, denoise_alpha)

        depth_for_vo = denoised if use_denoised and denoised is not None else depth

        # 2. VO Process
        vo_result = self.vo.process(depth_for_vo, conf, timestamp_ns)

        # 3. EKF Update (delegated)
        vo_pos = None
        vo_score = 0.0
        if vo_result:
            vo_pos = [vo_result["pos_x"], vo_result["pos_y"], vo_result["pos_z"]]
            vo_score = vo_result.get("vo_score", 0.0)

        fused_pos, euler = self.update_ekf(gyro_raw, accel_raw, dt, vo_pos, vo_score, timestamp_ns)

        return vo_result, fused_pos, euler, denoised
