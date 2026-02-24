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
        self.ekf.lateral_damping = config.get("lateral_damping", 0.1)

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
        u_prev = np.where(valid_z, f * pts_prev[:, 0] / pts_prev[:, 2] + cx, -1).astype(np.int32)
        v_prev = np.where(valid_z, f * pts_prev[:, 1] / pts_prev[:, 2] + cy, -1).astype(np.int32)
        
        in_bounds = valid_z & (u_prev >= 0) & (u_prev < W) & (v_prev >= 0) & (v_prev < H)
        new_denoised = self.denoised_depth.copy() # Actually maybe we should use the previous denoised as base? 
        # The previous implementation used self.denoised_depth.copy()
        
        u_v = u_idx[in_bounds]; v_v = v_idx[in_bounds]
        u_p = u_prev[in_bounds]; v_p = v_prev[in_bounds]
        
        prev_vals = self.denoised_depth[v_p, u_p]
        cur_vals = cur_depth_f[v_v, u_v]
        
        prev_valid = prev_vals > 0
        new_denoised[v_v[prev_valid], u_v[prev_valid]] = denoise_alpha * cur_vals[prev_valid] + (1 - denoise_alpha) * prev_vals[prev_valid]
        new_denoised[v_v[~prev_valid], u_v[~prev_valid]] = cur_vals[~prev_valid]
        
        u_nb = u_idx[~in_bounds]; v_nb = v_idx[~in_bounds]
        new_denoised[v_nb, u_nb] = cur_depth_f[v_nb, u_nb]
        # Checked viewer.py line 1294: new_denoised[v_nb, u_nb] = cur_depth_f[v_nb, u_nb]
        # I will fix it here.
        
        self.denoised_depth = new_denoised
        self.prev_ekf_pos = pos_cur.copy()
        self.prev_ekf_R = R_cur.copy()
        return self.denoised_depth
    
    def update_ekf(self, gyro_raw, accel_raw, dt, vo_pos=None, vo_score=0.0):
        # 1. IMU conversion and remapping
        gyro_rads = np.deg2rad(gyro_raw)
        accel_ms2 = accel_raw * 9.81
        if self.config.get("gyro_axis_map", "ZYX") == "ZYX":
            gyro_rads = gyro_rads[[2, 1, 0]]
            
        # 2. EKF Predict
        self.ekf.predict(gyro_rads, accel_ms2, dt)
        
        # 3. EKF Update if VO position provided
        if vo_pos is not None:
            min_conf = self.config.get("min_vo_confidence", 0.0)
            if vo_score >= min_conf:
                self.ekf.update(vo_pos, vo_score)
        
        return self.ekf.get_position()

    def process_frame(self, depth, conf, timestamp_ns, gyro_raw, accel_raw, dt, use_denoised=False, denoise_alpha=0.3):
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
            
        fused_pos = self.update_ekf(gyro_raw, accel_raw, dt, vo_pos, vo_score)
        
        return vo_result, fused_pos, denoised
