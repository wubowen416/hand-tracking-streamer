"""
Receives data from HTS, and send it to miracle human app.

Implement two functions:
1. Receive data from HTS via TCP.
2. Send data to miracle human.

These two functions run in 100 milliseconds interval.
The HTS should be faster than this, just throw that data away and only use the latest data.

"""

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

from mh.tcp_client import TCPClient as MHClient

import numpy as np

HOST = "127.0.0.1"
PORT = 8000


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    """Return a normalized quaternion."""
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    quat = _quat_normalize(quat)
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def _matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion (x, y, z, w)."""
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    return _quat_normalize(np.array([x, y, z, w], dtype=float))


def _matrix_to_euler(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to Euler angles in degrees, as [x, y, z].
    The euler is extrinsic, i.e., rotates around the global frame.

    Extrinsic ZXY (Unity's order): R = Ry(y) @ Rx(x) @ Rz(z).
    """
    sx = np.clip(-mat[1, 2], -1.0, 1.0)
    x = np.arcsin(sx)

    if abs(sx) < 1.0 - 1e-8:
        y = np.arctan2(mat[0, 2], mat[2, 2])
        z = np.arctan2(mat[1, 0], mat[1, 1])
    else:
        # Gimbal lock (x = +-90 deg): z is undetermined, so fix z = 0 and solve for y.
        y = np.arctan2(-mat[2, 0], mat[0, 0])
        z = 0.0

    return np.degrees(np.array([x, y, z], dtype=float))


@dataclass
class HandState:
    """State for a single hand."""

    wrist_position: Optional[np.ndarray] = None
    wrist_quat: Optional[np.ndarray] = None
    last_update: float = field(default_factory=time.monotonic)

    def update_wrist(self, data: Iterable[float]) -> None:
        """Update wrist pose from a 7-float sequence."""
        values = np.array(list(data), dtype=float)
        if values.size < 7:
            return
        # We use it in for MH in unity, so we use LH as is.
        # self.wrist_position = _convert_vec(values[:3])
        # self.wrist_quat = _convert_quat(values[3:7])
        self.wrist_position = values[:3]
        self.wrist_quat = values[3:7]
        self.last_update = time.monotonic()

    def wrist_point(self) -> Optional[np.ndarray]:
        """Return the wrist position if available."""
        return self.wrist_position
    
    def wrist_quaternion(self) -> Optional[np.ndarray]:
        """Return the wrist quaternion if available."""
        return self.wrist_quat
    

@dataclass
class HeadState:
    """State for the head."""

    position: Optional[np.ndarray] = None
    quat: Optional[np.ndarray] = None
    last_update: float = field(default_factory=time.monotonic)

    def update(self, data: Iterable[float]) -> None:
        """Update head pose from a 7-float sequence."""
        values = np.array(list(data), dtype=float)
        if values.size < 7:
            return
        self.position = values[:3]
        self.quat = values[3:7]
        self.last_update = time.monotonic()

    def head_point(self) -> Optional[np.ndarray]:
        """Return the head position if available."""
        return self.position
    
    def head_quat(self) -> Optional[np.ndarray]:
        """Return the head quaternion if available."""
        return self.quat


def _parse_line(line: str) -> Optional[Tuple[str, Tuple[float, ...]]]:
    """Parse a CSV wrist/head line into (key, floats)."""
    parts = [part.strip() for part in line.split(",")]
    if not parts:
        return None
    label = parts[0].lower()
    if "head" in label:
        key = "head"
    elif "wrist" in label and "right" in label:
        key = "right"
    elif "wrist" in label and "left" in label:
        key = "left"
    else:
        return None
    floats = []
    for part in parts[1:]:
        if not part:
            continue
        try:
            floats.append(float(part))
        except ValueError:
            continue
    return key, tuple(floats)


def make_homogeneous_matrix(
    rotation: np.ndarray = np.eye(3), translation: np.ndarray = np.zeros(3)
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a rotation and a translation."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def head_frame_matrix(head_position: np.ndarray, head_quat: np.ndarray) -> np.ndarray:
    """Build the 4x4 homogeneous matrix mapping the head-yaw frame to world space."""
    forward = _quat_to_matrix(head_quat) @ np.array([0.0, 0.0, 1.0])
    forward[1] = 0.0
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    xaxis = np.cross(up, forward)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(forward, xaxis)

    rotation = np.column_stack([xaxis, yaxis, forward])
    translation = np.array([head_position[0], 0.0, head_position[2]])
    return make_homogeneous_matrix(rotation, translation)


def world_to_head_frame(matrix: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Express a world-space position in the frame defined by `matrix`."""
    position_h = np.append(position, 1.0)
    return (np.linalg.inv(matrix) @ position_h)[:3]


def transform_position(matrix: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Transform a position using a 4x4 homogeneous matrix."""
    position_h = np.append(position, 1.0)
    return (matrix @ position_h)[:3]


HAND_HTS_TO_MH_RELABEL_MATRIX = {
    "right": np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    ),
    "left": np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    ),
}


def hand_wrist_quat_w_to_euler_mh(matrix_c2w: np.ndarray, wrist_quat_w: np.ndarray, side: str) -> np.ndarray:
    """Convert a world-space right-wrist quaternion into MH's wrist convention."""

    relabel_matrix = HAND_HTS_TO_MH_RELABEL_MATRIX.get(side.lower())
    assert relabel_matrix is not None, f"Invalid side: {side}. Must be 'right' or 'left'."

    r_wrist_w = make_homogeneous_matrix(rotation=_quat_to_matrix(wrist_quat_w))
    r_c = (np.linalg.inv(matrix_c2w) @ r_wrist_w)[:3, :3]
    r_mh = r_c @ relabel_matrix
    euler = _matrix_to_euler(r_mh)
    return euler


class StreamReceiver:
    """Background TCP receiver for HTS wrist data."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.right_hand = HandState()
        self.left_hand = HandState()
        self.head = HeadState()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._conn_threads: list[threading.Thread] = []

    def start(self) -> None:
        """Start the receiver thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the receiver thread and active connections."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for thread in list(self._conn_threads):
            if thread.is_alive():
                thread.join(timeout=0.5)

    def _handle_line(self, line: str) -> None:
        parsed = _parse_line(line)
        if not parsed:
            return
        side, floats = parsed
        if side == "right":
            self.right_hand.update_wrist(floats)
        elif side == "left":
            self.left_hand.update_wrist(floats)
        elif side == "head":
            self.head.update(floats)

    def _handle_conn(self, conn: socket.socket, addr) -> None:
        with conn:
            logging.info("Client connected: %s", addr)
            conn.settimeout(0.5)
            buffer = ""
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    logging.info("Client disconnected: %s", addr)
                    break
                try:
                    message = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                buffer += message
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line:
                        self._handle_line(line)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(5)
        sock.settimeout(0.5)
        logging.info("TCP server listening on %s:%d", self.host, self.port)
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = sock.accept()
                except socket.timeout:
                    continue
                thread = threading.Thread(
                    target=self._handle_conn, args=(conn, addr), daemon=True
                )
                self._conn_threads.append(thread)
                thread.start()
        finally:
            sock.close()


def make_mh_hand_msg(side, motion_frame, interval_s):
    data = {
        "id": "robot_hand_motion",
        "priority": 0,
        "isRelative": False,
        "isGloablCoordinates": True,
        "motionPartName": f"{side}HandController",
        "thoughSafetyStopoverPoint": False,
        "motionHandData": [
            {
                "id": "robot_hand_motion",
                "motionTowardObject": "BodyController",  # Spine, seems the same as BodyController
                "targetMotionMode": 1,
                "transitionCoordinate": 0,
                "targetPoint": {
                    "x": motion_frame[f"{side}_hand_cartesian_pos"][0],
                    "y": motion_frame[f"{side}_hand_cartesian_pos"][1],
                    "z": motion_frame[f"{side}_hand_cartesian_pos"][2],
                },
                "translateSpeed": -1,
                "translateTime": interval_s * 1000,
                "rotationCoordinate": 9,
                "targetRotation": {
                    "x": motion_frame[f"{side}_hand_euler_angle"][0],
                    "y": motion_frame[f"{side}_hand_euler_angle"][1],
                    "z": motion_frame[f"{side}_hand_euler_angle"][2],
                },
                "rotateSpeed": -1,
                "rotateTime": interval_s * 1000,
                "keepTime": 0,
                "mode": 2,
                "gazeTracking": True,
                "priority": 0,
                "tracking": True,
                "fingerData": [
                    {
                        "motionPartName": "RightFingers",
                        "targetAngle": 0,
                        "springValue": 10,
                    },
                    {
                        "motionPartName": "LeftFingers",
                        "targetAngle": 0,
                        "springValue": 10,
                    },
                ],
            }
        ],
    }
    msg = "playthishandmotion={}".format(json.dumps(data))
    return msg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    receiver = StreamReceiver(HOST, PORT)
    receiver.start()

    sender = MHClient("127.0.0.1", 21000)
    sender.connect()
    sender.send_message("playmotion=righthandbaseposition")
    sender.send_message("playmotion=lefthandbaseposition")

    interval_s = 0.1

    # Scale and offset for alignment
    y_scale = 1.6 / 1.7  # 1.6 = 1.7 - 0.1, where 0.1 is about the distance from head to eye, 1.7 is the height of the person

    try:
        while True:
            # Compute head frame matrix for alignment
            head_pos = receiver.head.head_point()
            head_quat = receiver.head.head_quat()
            if head_pos is None or head_quat is None:
                continue
            matrix_c2w = head_frame_matrix(head_pos, head_quat)

            # Process right hand data
            right_hand_pos_w = receiver.right_hand.wrist_point()
            right_hand_quat_w = receiver.right_hand.wrist_quaternion()
            if right_hand_pos_w is not None and right_hand_quat_w is not None:
                right_hand_pos_c = transform_position(np.linalg.inv(matrix_c2w), right_hand_pos_w)
                right_hand_euler_mh = hand_wrist_quat_w_to_euler_mh(matrix_c2w, right_hand_quat_w, "right")

                # Scale
                right_hand_pos_c[1] *= y_scale

                # print("right hand pos:", right_hand_pos_c)
                print("right hand euler (MH frame):", right_hand_euler_mh)
                
                right_motion_frame = {
                    "Right_hand_cartesian_pos": right_hand_pos_c.tolist(),
                    "Right_hand_euler_angle": right_hand_euler_mh.tolist(),
                }
                right_hand_msg = make_mh_hand_msg("Right", right_motion_frame, interval_s)
                sender.send_message(right_hand_msg)

                # print("Sent right hand message:", right_hand_msg)

            # Process left hand data
            left_hand_pos_w = receiver.left_hand.wrist_point()
            left_hand_quat_w = receiver.left_hand.wrist_quaternion()
            if left_hand_pos_w is not None and left_hand_quat_w is not None:
                left_hand_pos_c = transform_position(np.linalg.inv(matrix_c2w), left_hand_pos_w)
                left_hand_euler_mh = hand_wrist_quat_w_to_euler_mh(matrix_c2w, left_hand_quat_w, "left")

                # Scale
                left_hand_pos_c[1] *= y_scale

                print("left hand euler (MH frame):", left_hand_euler_mh)
                
                left_motion_frame = {
                    "Left_hand_cartesian_pos": left_hand_pos_c.tolist(),
                    "Left_hand_euler_angle": left_hand_euler_mh.tolist(),
                }
                left_hand_msg = make_mh_hand_msg("Left", left_motion_frame, interval_s)
                sender.send_message(left_hand_msg)

            time.sleep(interval_s)

    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
