from PySide6.QtCore import QThread, Signal, QObject
import numpy as np

class ProcessingWorker(QObject):
    finished = Signal(list)  # Returns list of results
    progress = Signal(int)
    
    def __init__(self, engine, frame_loader, imu_interpolated, start_idx, end_idx):
        super().__init__()
        self.engine = engine
        self.frame_loader = frame_loader
        self.imu_interpolated = imu_interpolated
        self.start_idx = start_idx
        self.end_idx = end_idx
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        results = []
        self.engine.reset() # Should we reset? usually yes for full re-runs
        
        for i in range(self.start_idx, self.end_idx + 1):
            if self._abort:
                break
                
            frame = self.frame_loader.get_frame(i)
            if frame is None:
                results.append(None)
                continue
                
            # Extract IMU data for this frame
            gyro = np.array([
                self.imu_interpolated["gyro_x"][i],
                self.imu_interpolated["gyro_y"][i],
                self.imu_interpolated["gyro_z"][i]
            ])
            accel = np.array([
                self.imu_interpolated["acc_x"][i],
                self.imu_interpolated["acc_y"][i],
                self.imu_interpolated["acc_z"][i]
            ])
            
            # Simplified dt for re-processing or use timestamps?
            # Viewer uses: ts - prev_ts
            ts = frame["timestamp_ns"]
            prev_ts = self.frame_loader.get_frame(i-1)["timestamp_ns"] if i > 0 else None
            dt = (ts - prev_ts) / 1e9 if prev_ts else 0.033
            
            vo_res, fused_pos, euler, speeds, denoised = self.engine.process_frame(
                frame["depth"], frame["conf"], ts,
                gyro, accel, dt,
                use_denoised=self.engine.config.get("vo_use_denoised", False),
                denoise_alpha=self.engine.config.get("denoise_alpha", 0.3)
            )
            
            results.append((vo_res, fused_pos, euler, speeds))
            self.progress.emit(i)
            
        self.engine.stop_logging()
        self.finished.emit(results)
