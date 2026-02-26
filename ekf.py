import numpy as np


class ExtendedKalmanFilter:
    """6-state EKF: position (x, y, z) + velocity (vx, vy, vz).

    Orientation is tracked separately via quaternion integration and used only
    to rotate IMU accelerations into the world frame before feeding the EKF.
    """

    def __init__(self):
        # State: [px, py, pz, vx, vy, vz]
        self.state = np.zeros(6)
        self.P = np.eye(6) * 0.1

        # Orientation quaternion [w, x, y, z] (not part of state vector)
        # Initial: camera Z (forward) -> world Y (forward), -90° around X
        # q = [cos(-45°), sin(-45°), 0, 0] = [√2/2, -√2/2, 0, 0]
        self.q = np.array([0.70710678, -0.70710678, 0.0, 0.0])

        self.up_direction = np.array([0.0, 0.0, 1.0])

        self.fixed_height_enabled = False
        self.fixed_height = 1.2
        self.height_strength = 0.5

        self.process_noise = 0.1
        self.measurement_noise = 1.0

        # Wheelchair motion model constraints
        self.max_speed = 2.0  # m/s — 0 = disabled
        self.max_rotation_speed = 1.5  # rad/s — 0 = disabled
        self.lateral_damping = 0.1  # 0 = no lateral motion, 1 = unconstrained

        self.prev_timestamp = None
        self.last_dt = 0.033

        self.last_gyro = np.zeros(3)
        self.last_accel = np.zeros(3)

    def reset(self):
        self.state = np.zeros(6)
        self.P = np.eye(6) * 0.1
        # Reset to initial orientation: camera Z (forward) -> world Y (forward)
        self.q = np.array([0.70710678, -0.70710678, 0.0, 0.0])
        self.prev_timestamp = None
        self.last_dt = 0.033
        self.last_gyro = np.zeros(3)
        self.last_accel = np.zeros(3)

    def detect_up_direction(self, imu_data, samples=100):
        if imu_data is None or len(imu_data) < 10:
            self.up_direction = np.array([0.0, 0.0, 1.0])
            return

        accel_data = imu_data[:, 1:4]
        n = min(samples, len(accel_data))
        mean_accel = np.mean(accel_data[:n], axis=0)
        norm = np.linalg.norm(mean_accel)
        if norm > 1e-6:
            self.up_direction = mean_accel / norm
            # Optionally reset initial orientation so this up_direction points to world Z
            self._init_q_from_up()
        else:
            self.up_direction = np.array([0.0, 0.0, 1.0])

    def _init_q_from_up(self):
        """Initialize quaternion self.q such that self.up_direction aligns with world Z-up.

        Standard camera orientation (Z forward) is still rotated to world Y forward.
        """
        # 1. Align IMU 'up_direction' with World Z [0, 0, 1]
        v_from = self.up_direction
        v_to = np.array([0.0, 0.0, 1.0])

        # Rotation between two vectors: q = [1 + dot(v1,v2), cross(v1,v2)]
        dot = np.dot(v_from, v_to)
        if dot < -0.999999:
            # Opposite vectors
            self.q = np.array([0.0, 1.0, 0.0, 0.0]) # 180 around X
        else:
            cross = np.cross(v_from, v_to)
            s = np.sqrt((1 + dot) * 2)
            self.q = np.array([s / 2, cross[0] / s, cross[1] / s, cross[2] / s])

        self._normalize_quaternion()

        # 2. Add the -90 degree rotation around camera-X (now in world space roughly)
        # to point camera-Z (pinhole forward) to world-Y (forward).
        # We rotate by -pi/2 around World X [1, 0, 0]
        theta = -np.pi / 2
        q_rot = np.array([np.cos(theta / 2), np.sin(theta / 2), 0, 0])
        self.q = self._quaternion_multiply(q_rot, self.q)
        self._normalize_quaternion()

    def set_fixed_height(self, enabled, height_meters=1.2, strength=0.5):
        self.fixed_height_enabled = enabled
        self.fixed_height = height_meters
        self.height_strength = strength

    def set_noise_params(self, process_noise=0.1, measurement_noise=1.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def _quaternion_rotate(self, q, v):
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        result = self._quaternion_multiply(self._quaternion_multiply(q, v_quat), q_conj)
        return result[1:4]

    def _normalize_quaternion(self):
        norm = np.linalg.norm(self.q)
        if norm > 0:
            self.q = self.q / norm

    def _apply_motion_constraints(self):
        """Enforce wheelchair kinematic constraints on the velocity state.

        1. Max speed clamp: limits ||velocity|| to max_speed.
        2. Lateral damping: suppresses sideways velocity relative to the camera's
           forward direction (non-holonomic constraint for a wheeled vehicle).
           Camera +Z is forward in the camera frame (pinhole convention).
           With Z=up in world frame, the floor plane is XY.
        """
        if self.max_speed > 0:
            vel = self.state[3:6]
            speed = np.linalg.norm(vel)
            if speed > self.max_speed:
                self.state[3:6] = vel * (self.max_speed / speed)

        if self.lateral_damping < 1.0:
            # Camera forward = [0, 0, 1] in camera frame, rotated to world frame
            forward_world = self._quaternion_rotate(self.q, np.array([0.0, 0.0, 1.0]))
            # Project onto the floor plane (Z=up → XY is floor)
            forward_floor = np.array([forward_world[0], forward_world[1]])
            fn = np.linalg.norm(forward_floor)
            if fn > 1e-6:
                forward_floor /= fn
                vel_floor = self.state[3:5]
                v_fwd = np.dot(vel_floor, forward_floor) * forward_floor
                v_lat = vel_floor - v_fwd
                self.state[3:5] = v_fwd + v_lat * self.lateral_damping

    def predict(self, gyro, accel, dt):
        if dt <= 0 or dt > 0.5:
            return

        # Clamp angular velocity to max rotation speed
        if self.max_rotation_speed > 0:
            gyro_norm = np.linalg.norm(gyro)
            if gyro_norm > self.max_rotation_speed:
                gyro = gyro * (self.max_rotation_speed / gyro_norm)

        # --- Quaternion integration (q = [w, x, y, z]) ---
        q = self.q
        omega = gyro
        # Correct quaternion kinematics: q_dot = 0.5 * Omega(omega) * q
        q_dot = 0.5 * np.array(
            [
                -omega[0] * q[1] - omega[1] * q[2] - omega[2] * q[3],
                omega[0] * q[0] - omega[1] * q[3] + omega[2] * q[2],
                omega[0] * q[3] + omega[1] * q[0] - omega[2] * q[1],
                -omega[0] * q[2] + omega[1] * q[1] + omega[2] * q[0],
            ]
        )
        self.q = self.q + q_dot * dt
        self._normalize_quaternion()

        # --- Rotate accel into world frame and remove gravity ---
        accel_world = self._quaternion_rotate(self.q, accel)
        # In world frame, gravity is ALWAYS [0, 0, 9.81] if world Z is up
        gravity = np.array([0.0, 0.0, 9.81])
        accel_corrected = accel_world - gravity

        # --- State transition: constant-acceleration kinematics ---
        ax, ay, az = accel_corrected
        # pos += vel*dt + 0.5*acc*dt²
        self.state[0] += self.state[3] * dt + 0.5 * ax * dt * dt
        self.state[1] += self.state[4] * dt + 0.5 * ay * dt * dt
        self.state[2] += self.state[5] * dt + 0.5 * az * dt * dt
        # vel += acc*dt
        self.state[3] += ax * dt
        self.state[4] += ay * dt
        self.state[5] += az * dt

        # --- Covariance propagation with Jacobian F ---
        # F = d(f)/d(state) for [p, v]:
        #   p_new = p + v*dt  =>  dp_new/dp=I, dp_new/dv=I*dt
        #   v_new = v + a*dt  =>  dv_new/dp=0, dv_new/dv=I
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        Q = np.eye(6) * self.process_noise
        Q[0:3, 0:3] *= dt * dt
        Q[3:6, 3:6] *= dt * dt

        self.P = F @ self.P @ F.T + Q

        self.last_gyro = gyro
        self.last_accel = accel
        self.last_dt = dt

        if self.fixed_height_enabled:
            current_z = self.state[2]
            # Pull Z back to fixed_height based on strength
            self.state[2] = (
                current_z * (1 - self.height_strength)
                + self.fixed_height * self.height_strength
            )
            # Also zero out vertical velocity to prevent continued "falling"
            current_vz = self.state[5]
            self.state[5] = current_vz * (1 - self.height_strength)

        self._apply_motion_constraints()

    def update(self, vo_position, vo_confidence=1.0):
        if vo_position is None:
            return

        measured = np.array(vo_position)

        r_scalar = self.measurement_noise / (vo_confidence + 0.01)
        R_matrix = np.eye(3) * r_scalar

        # Observation matrix: we observe position directly
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        z = measured
        z_pred = self.state[0:3]

        y = z - z_pred

        # Innovation gate: reject VO measurements that imply physically impossible motion.
        # Maximum plausible position change = max_speed × dt × generous_margin (5×).
        if self.max_speed > 0 and self.last_dt > 0:
            max_plausible = self.max_speed * self.last_dt * 5.0
            if np.linalg.norm(y) > max_plausible:
                return  # discard this measurement — too large to be real

        S = H @ self.P @ H.T + R_matrix
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y

        # Joseph form for numerical stability
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_matrix @ K.T


        self._apply_motion_constraints()

    def get_position(self):
        return self.state[0], self.state[1], self.state[2]

    def get_velocity(self):
        return self.state[3], self.state[4], self.state[5]

    def get_rotation_matrix(self):
        """Convert the internal quaternion [w, x, y, z] to a 3x3 rotation matrix."""
        w, x, y, z = self.q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def get_euler_angles(self):
        """Convert the internal quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw) in degrees.
        Uses ZYX convention (yaw, then pitch, then roll).
        """
        w, x, y, z = self.q
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    def get_speeds(self):
        """Get (forward_speed, lateral_speed) in m/s relative to camera orientation."""
        vel = self.state[3:6]
        # Camera forward = [0, 0, 1] in camera frame, rotated to world frame
        forward_world = self._quaternion_rotate(self.q, np.array([0.0, 0.0, 1.0]))
        # Project onto the floor plane (Z=up → XY is floor)
        forward_floor = np.array([forward_world[0], forward_world[1], 0.0])
        fn = np.linalg.norm(forward_floor)
        if fn > 1e-6:
            forward_floor /= fn
            fwd_speed = np.dot(vel, forward_floor)
            # Lateral is relative to forward on XY plane
            # [fx, fy, 0] rotated 90 deg around Z is [-fy, fx, 0]
            lat_dir = np.array([-forward_floor[1], forward_floor[0], 0.0])
            lat_speed = np.dot(vel, lat_dir)
            return fwd_speed, lat_speed
        return 0.0, 0.0
