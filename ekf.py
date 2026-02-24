import numpy as np


class ExtendedKalmanFilter:
    def __init__(self):
        self.state = np.zeros(10)
        self.state[9] = 1.0

        self.P = np.eye(10) * 0.1

        self.q = np.array([0, 0, 0, 1.0])

        self.up_direction = np.array([0, 0, 1.0])

        self.fixed_height_enabled = False
        self.fixed_height = 1.2
        self.height_strength = 0.5

        self.process_noise = 0.1
        self.measurement_noise = 1.0

        self.prev_timestamp = None

        self.last_gyro = np.zeros(3)
        self.last_accel = np.zeros(3)

    def reset(self):
        self.state = np.zeros(10)
        self.state[9] = 1.0
        self.P = np.eye(10) * 0.1
        self.q = np.array([0, 0, 0, 1.0])
        self.prev_timestamp = None
        self.last_gyro = np.zeros(3)
        self.last_accel = np.zeros(3)

    def detect_up_direction(self, imu_data, samples=100):
        if imu_data is None or len(imu_data) < 10:
            self.up_direction = np.array([0, 0, 1.0])
            return

        accel_data = imu_data[:, 1:4]
        n = min(samples, len(accel_data))
        mean_accel = np.mean(accel_data[:n], axis=0)
        norm = np.linalg.norm(mean_accel)
        if norm > 0:
            self.up_direction = -mean_accel / norm
        else:
            self.up_direction = np.array([0, 0, 1.0])

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
        v_quat = np.array([0, v[0], v[1], v[2]])
        result = self._quaternion_multiply(self._quaternion_multiply(q, v_quat), q_conj)
        return result[1:4]

    def _normalize_quaternion(self):
        norm = np.linalg.norm(self.q)
        if norm > 0:
            self.q = self.q / norm

    def predict(self, gyro, accel, dt):
        if dt <= 0 or dt > 0.5:
            return

        q = self.q

        omega = gyro
        q_dot = 0.5 * np.array(
            [
                -omega[0] * q[1] - omega[1] * q[2] - omega[2] * q[3],
                omega[0] * q[0] + omega[2] * q[2] - omega[1] * q[3],
                -omega[0] * q[2] + omega[1] * q[0] + omega[2] * q[1],
                omega[0] * q[3] + omega[1] * q[2] - omega[2] * q[1],
            ]
        )

        self.q = self.q + q_dot * dt
        self._normalize_quaternion()

        accel_world = self._quaternion_rotate(self.q, accel)

        gravity = 9.81 * self.up_direction
        accel_corrected = accel_world - gravity

        self.state[0] += self.state[3] * dt + 0.5 * accel_corrected[0] * dt * dt
        self.state[1] += self.state[4] * dt + 0.5 * accel_corrected[1] * dt * dt
        self.state[2] += self.state[5] * dt + 0.5 * accel_corrected[2] * dt * dt

        self.state[3] += accel_corrected[0] * dt
        self.state[4] += accel_corrected[1] * dt
        self.state[5] += accel_corrected[2] * dt

        Q = np.eye(10) * self.process_noise
        Q[0:3, 0:3] *= dt * dt
        Q[3:6, 3:6] *= dt * dt

        self.P = self.P + Q

        self.last_gyro = gyro
        self.last_accel = accel

    def update(self, vo_position, vo_confidence=1.0):
        if vo_position is None:
            return

        measured = np.array(vo_position)

        R = self.measurement_noise / (vo_confidence + 0.01)

        H = np.zeros((3, 10))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        z = measured
        z_pred = self.state[0:3]

        y = z - z_pred

        S = H @ self.P @ H.T + np.eye(3) * R

        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y

        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        if self.fixed_height_enabled:
            z_idx = 2
            current_z = self.state[z_idx]
            constrained_z = (
                current_z * (1 - self.height_strength)
                + self.fixed_height * self.height_strength
            )
            self.state[z_idx] = constrained_z

    def get_position(self):
        return self.state[0], self.state[1], self.state[2]

    def get_velocity(self):
        return self.state[3], self.state[4], self.state[5]
