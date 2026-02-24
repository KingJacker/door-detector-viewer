import cv2
import numpy as np
import time

class VisualOdometry:
    def __init__(self, config):
        self.config = config

        # Increase this if you want more robust tracking
        self.target_features = 200 
        self.min_features = 50     

        self.width = 240
        self.height = 180

        self._setup_camera()
        self._setup_detector()
        self._setup_clahe()

        self.prev_gray = None
        self.prev_pts = None

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.R_world = np.eye(3)

        self.tracked_count = 0
        self.rejected_count = 0
        self.ratio = 0.0
        self.mean_depth = 0.0
        self.mean_conf = 0.0
        self.fps = 0.0

        self._frame_times = []

    def _setup_camera(self):
        f_mm = self.config.get("focal_length_mm", 3.4)
        # Handle cases where sensor_size might be missing
        sensor_size = self.config.get("sensor_size", "1/6")
        sensor_width = self._sensor_width_mm(sensor_size)

        self.focal_length = (f_mm / sensor_width) * self.width
        self.center = (self.width / 2.0, self.height / 2.0)

    def _sensor_diagonal_mm(self, sensor_size):
        sizes = {
            "1/6": 3.0, "1/4": 4.0, "1/3": 6.0,
            "1/2.7": 7.0, "1/2": 8.0, "2/3": 11.0, "1": 16.0,
        }
        return sizes.get(sensor_size, 3.0)

    def _sensor_width_mm(self, sensor_size):
        aspect = 4 / 3
        diag = self._sensor_diagonal_mm(sensor_size)
        return diag / np.sqrt(1 + aspect**2)

    def _setup_detector(self):
        # FAST detector is good for speed
        threshold = self.config.get("detector_threshold", 20)
        self.detector = cv2.FastFeatureDetector_create(threshold=int(threshold))

        self.lk_params = dict(
            winSize=(21, 21),  # Slightly larger window for depth maps
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.quality_threshold = self.config.get("rejection_threshold", 8.0)
        self.stationary_threshold = self.config.get("stationary_threshold", 0.5)

    def _setup_clahe(self):
        enabled = self.config.get("clahe_enabled", False)
        clip_limit = self.config.get("clahe_clip_limit", 2.0)
        grid_size = int(self.config.get("clahe_grid_size", 8))

        if enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
            )
        else:
            self.clahe = None

    def update_config(self, config):
        self.config = config
        self._setup_detector()
        self._setup_clahe()
        self._setup_camera()

    def process(self, depth, conf, timestamp_ns=None):
        start_time = time.time()

        conf_threshold = self.config.get("conf_threshold", 100)
        
        # 1. Prepare Masks
        if conf is not None:
            mask_conf = (conf >= conf_threshold).astype(np.uint8) * 255
            masked_depth = np.where(conf >= conf_threshold, depth, 0)
        else:
            mask_conf = np.ones_like(depth, dtype=np.uint8) * 255
            masked_depth = depth

        # 2. Preprocess Image (Depth -> Grayscale uint8)
        gray = self._preprocess(masked_depth)

        result = {
            "tracked": 0, "rejected": 0, "ratio": 0.0,
            "pos_x": float(self.position[0]), 
            "pos_y": float(self.position[1]), 
            "pos_z": float(self.position[2]),
            "mean_depth": 0.0, "mean_conf": 0.0, "fps": 0.0,
            "tracked_pts": [], "rejected_pts": []
        }

        # Stats
        valid_mask = masked_depth > 0
        if valid_mask.any():
            result["mean_depth"] = float(masked_depth[valid_mask].mean())
        if conf is not None and valid_mask.any():
            result["mean_conf"] = float(conf[valid_mask].mean())

        # 3. Initialization (First Frame)
        if self.prev_pts is None:
            kp = self.detector.detect(gray, mask_conf)
            if kp:
                self.prev_pts = cv2.KeyPoint_convert(kp).reshape(-1, 1, 2)
            else:
                self.prev_pts = np.empty((0, 1, 2), dtype=np.float32)
            
            self.prev_gray = gray
            result["tracked"] = len(self.prev_pts)
            return result

        # 4. Optical Flow Tracking
        if len(self.prev_pts) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            )
        else:
            p1, st, err = None, None, None

        # 5. Filter Bad Points
        if p1 is not None:
            good_indices = (st == 1).flatten()
            if err is not None:
                # Filter by tracking error (optical flow consistency)
                good_indices = good_indices & (err.flatten() <= self.quality_threshold)
            
            # Ensure points stay within image bounds
            h, w = gray.shape
            for i, pt in enumerate(p1):
                if good_indices[i]:
                    x, y = pt.flatten()
                    if not (0 <= x < w and 0 <= y < h):
                        good_indices[i] = 0

            good_new = p1[good_indices]
            good_old = self.prev_pts[good_indices]
            
            # Update stats
            total_tracked = len(self.prev_pts)
            valid_tracked = len(good_new)
            self.rejected_count = total_tracked - valid_tracked
            self.ratio = (valid_tracked / total_tracked * 100) if total_tracked > 0 else 0.0

            result["tracked"] = valid_tracked
            result["rejected"] = self.rejected_count
            result["ratio"] = self.ratio
            result["tracked_pts"] = good_new.reshape(-1, 2).tolist()
            
            # 6. Pose Estimation (Only if we have enough points)
            if valid_tracked > 6: # Need min 5 for finding Essential Matrix
                try:
                    # Calculate Essential Matrix
                    E, mask_pose = cv2.findEssentialMat(
                        good_old, good_new, 
                        focal=self.focal_length, pp=self.center,
                        method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )

                    if E is not None and E.shape == (3, 3):
                        _, R, t, _ = cv2.recoverPose(
                            E, good_old, good_new, 
                            focal=self.focal_length, pp=self.center
                        )

                        # Scale Recovery using Depth
                        # We look up the depth of the tracked points to scale the unit vector 't'
                        depths = []
                        for pt in good_old: # Use old points for depth lookup (matches prev frame)
                            x, y = int(pt[0][0]), int(pt[0][1])
                            # Check bounds again just to be safe
                            if 0 <= x < self.width and 0 <= y < self.height:
                                # We need depth from the PREVIOUS frame for strictly correct math,
                                # but using current masked_depth is a common approximation.
                                # Ideally, cache prev_depth. For now, use current.
                                d = masked_depth[y, x] 
                                if d > 100: # Ignore noise/0 depth
                                    depths.append(d)

                        if depths:
                            median_depth = np.median(depths)
                            # Simple scale estimation
                            t_scaled = t * median_depth
                            
                            # Update Global Position
                            # pos = pos + R_current * t_scaled
                            self.position += (self.R_world @ t_scaled).flatten()
                            self.R_world = self.R_world @ R

                except Exception as e:
                    # print(f"VO Error: {e}") 
                    pass

        else:
            good_new = np.empty((0, 1, 2), dtype=np.float32)

        result["pos_x"] = float(self.position[0])
        result["pos_y"] = float(self.position[1])
        result["pos_z"] = float(self.position[2])

        # 7. Replenishment (Crucial Fix)
        # Instead of resetting when < 10, we add points if < target
        if len(good_new) < self.target_features:
            # Mask out existing points so we don't redetect them
            mask_replenish = mask_conf.copy()
            for pt in good_new:
                cv2.circle(mask_replenish, (int(pt[0][0]), int(pt[0][1])), 8, 0, -1)
            
            # Detect new features in empty areas
            kp_new = self.detector.detect(gray, mask_replenish)
            
            if kp_new:
                pts_new = cv2.KeyPoint_convert(kp_new).reshape(-1, 1, 2)
                # Stack existing tracked points with new points
                if len(good_new) > 0:
                    good_new = np.vstack((good_new, pts_new))
                else:
                    good_new = pts_new

        # 8. Update State for Next Frame
        self.prev_gray = gray
        self.prev_pts = good_new

        # FPS Calculation
        if timestamp_ns:
            if self._frame_times:
                dt = (timestamp_ns - self._frame_times[-1]) / 1e9
                if dt > 0:
                    current_fps = 1.0 / dt
                    self.fps = 0.9 * self.fps + 0.1 * current_fps
            
            self._frame_times.append(timestamp_ns)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)

        result["fps"] = self.fps

        return result

    def _preprocess(self, depth):
        depth_f = depth.astype(np.float32)

        # Better normalization strategy for VO:
        # Avoid cv2.NORM_MINMAX on every frame because it causes flickering
        # if the min/max depth in the scene changes.
        # Fixed scaling is better. Assuming max range ~4000mm.
        
        # Apply CLAHE if enabled
        if self.clahe is not None:
            # Normalize reasonably to 0-255 for CLAHE
            # detailed structure is more important than absolute value here
            gray = cv2.normalize(depth_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            gray = self.clahe.apply(gray)
            return gray
        
        # Fallback if CLAHE is off
        normalized = cv2.normalize(depth_f, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.R_world = np.eye(3)
        self.tracked_count = 0
        self.rejected_count = 0
        self.ratio = 0.0
        self.mean_depth = 0.0
        self.mean_conf = 0.0
        self.fps = 0.0
        self._frame_times = []