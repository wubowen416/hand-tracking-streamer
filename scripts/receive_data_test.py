"""
Receives data from HTS, and send it to miracle human app.

Implement two functions:
1. Receive data from HTS via TCP.
2. Send data to miracle human.

These two functions run in 100 milliseconds interval.
The HTS should be faster than this, just throw that data away and only use the latest data.

"""

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

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


# Unity LH (x right, y up, z forward) -> RH (x front, y left, z up)
_UNITY_TO_RH = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


def _convert_vec(vec: np.ndarray) -> np.ndarray:
    """Convert a vector from Unity LH to RH coordinates."""
    return _UNITY_TO_RH @ vec


def _convert_quat(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion from Unity LH to RH coordinates."""
    r_unity = _quat_to_matrix(quat)
    r_rh = _UNITY_TO_RH @ r_unity @ _UNITY_TO_RH.T
    return _matrix_to_quat(r_rh)


@dataclass
class HandState:
    """State for a single hand."""

    side: str
    wrist_position: Optional[np.ndarray] = None
    wrist_quat: Optional[np.ndarray] = None
    last_update: float = field(default_factory=time.monotonic)

    def update_wrist(self, data: Iterable[float]) -> None:
        """Update wrist pose from a 7-float sequence."""
        values = np.array(list(data), dtype=float)
        if values.size < 7:
            return
        self.wrist_position = _convert_vec(values[:3])
        self.wrist_quat = _convert_quat(values[3:7])
        self.last_update = time.monotonic()

    def wrist_point(self) -> Optional[np.ndarray]:
        """Return the wrist position if available."""
        return self.wrist_position


def _parse_line(line: str) -> Optional[Tuple[str, Tuple[float, ...]]]:
    """Parse a CSV wrist line into (side, floats)."""
    parts = [part.strip() for part in line.split(",")]
    if not parts:
        return None
    label = parts[0].lower()
    if "wrist" not in label:
        return None
    side = "right" if "right" in label else "left" if "left" in label else ""
    if not side:
        return None
    floats = []
    for part in parts[1:]:
        if not part:
            continue
        try:
            floats.append(float(part))
        except ValueError:
            continue
    return side, tuple(floats)


class StreamReceiver:
    """Background TCP receiver for HTS wrist data."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.hands: Dict[str, HandState] = {
            "right": HandState("right"),
            "left": HandState("left"),
        }
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start the receiver thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the receiver thread."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _handle_line(self, line: str) -> None:
        parsed = _parse_line(line)
        if not parsed:
            return
        side, floats = parsed
        self.hands[side].update_wrist(floats)

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
                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line:
                        self._handle_line(line)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        sock.settimeout(0.5)
        logging.info("TCP server listening on %s:%d", self.host, self.port)
        try:
            while not self._stop.is_set():
                logging.info("Waiting for connection...")
                conn = None
                addr = None
                while not self._stop.is_set():
                    try:
                        conn, addr = sock.accept()
                        break
                    except socket.timeout:
                        continue
                if conn is None:
                    break
                self._handle_conn(conn, addr)
        finally:
            sock.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    receiver = StreamReceiver(HOST, PORT)
    receiver.start()

    try:
        while True:
            print(
                {
                    "right": receiver.hands["right"].wrist_point(),
                    "left": receiver.hands["left"].wrist_point(),
                }
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
