"""Direct single-drone Tello control and camera adapters.

This mirrors the known-good single-drone scripts that use djitellopy's
BackgroundFrameRead instead of the swarm video demuxer.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import cv2
import numpy as np

from swarm_camera import CameraSnapshot


StatusCallback = Callable[[str], None]


@dataclass
class DirectTelloLink:
    ip: str
    tello: object | None = None
    frame_read: object | None = None
    connected: bool = False
    streaming: bool = False

    def connect(self) -> object:
        if self.connected and self.tello is not None:
            return self.tello

        try:
            from djitellopy import Tello
        except ModuleNotFoundError as error:
            raise RuntimeError("djitellopy is required for direct single-drone video") from error

        tello = Tello(host=self.ip)
        tello.connect(wait_for_state=False)
        self.tello = tello
        self.connected = True
        return tello

    def stop_stream(self) -> None:
        frame_read = self.frame_read
        if frame_read is not None:
            try:
                frame_read.stop()
            except Exception:
                pass
            self.frame_read = None

        tello = self.tello
        if tello is not None and self.streaming:
            try:
                tello.streamoff()
            except Exception:
                pass
        self.streaming = False

    def close(self) -> None:
        self.stop_stream()
        tello = self.tello
        if tello is not None:
            try:
                tello.end()
            except Exception:
                pass
        self.tello = None
        self.connected = False


class DirectTelloController:
    """Small controller adapter matching the subset used by SingleDroneSearchRunner."""

    def __init__(self, link: DirectTelloLink) -> None:
        self.link = link
        self.ips = [link.ip]
        self._airborne = False

    def check_all_ready(
        self,
        retries: int = 2,
        min_battery: int = 20,
        status: StatusCallback | None = None,
        **_kwargs,
    ) -> bool:
        for attempt in range(1, retries + 1):
            try:
                tello = self.link.connect()
                battery = int(tello.query_battery())
                self._status(status, f"{self.link.ip}: battery={battery}%")
                return battery >= min_battery
            except Exception as error:
                self._status(status, f"{self.link.ip}: preflight attempt {attempt}/{retries} failed: {error}")
                time.sleep(0.4)
        return False

    def takeoff_sequential(self, settle_seconds: float = 3.0, status: StatusCallback | None = None, **_kwargs) -> bool:
        try:
            tello = self.link.connect()
            self._status(status, f"{self.link.ip}: takeoff")
            tello.takeoff()
            self._airborne = True
            if settle_seconds > 0:
                time.sleep(settle_seconds)
            return True
        except Exception as error:
            self._status(status, f"{self.link.ip}: takeoff failed: {error}")
            return False

    def send_single_command(
        self,
        ip: str,
        command: str,
        status: StatusCallback | None = None,
        **_kwargs,
    ) -> bool:
        if ip != self.link.ip:
            return False
        try:
            tello = self.link.connect()
            self._status(status, f"{ip}: {command}")
            tello.send_control_command(command)
            return True
        except Exception as error:
            self._status(status, f"{ip}: {command} failed: {error}")
            return False

    def land_all(self, status: StatusCallback | None = None, **_kwargs) -> bool:
        try:
            tello = self.link.connect()
            self._status(status, f"{self.link.ip}: land")
            tello.land()
            self._airborne = False
            return True
        except Exception as error:
            self._status(status, f"{self.link.ip}: land failed: {error}")
            return False

    def any_airborne(self) -> bool:
        return self._airborne

    def close(self) -> None:
        self.link.close()

    @staticmethod
    def _status(status: StatusCallback | None, message: str) -> None:
        if status is not None:
            status(message)


class DirectTelloCamera:
    """Direct one-drone camera adapter backed by djitellopy BackgroundFrameRead."""

    def __init__(self, link: DirectTelloLink, frame_timeout: float = 10.0) -> None:
        self.link = link
        self.frame_timeout = frame_timeout
        self.version = 0
        self.last_frame_at = 0.0
        self._running = False
        self._last_frame: np.ndarray | None = None

    def start(self) -> None:
        self.stop()
        tello = self.link.connect()
        try:
            tello.streamoff()
        except Exception:
            pass
        tello.streamon()
        self.link.streaming = True
        self.link.frame_read = tello.get_frame_read()
        self._wait_for_first_frame()
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.link.stop_stream()

    def get_snapshot(self, copy: bool = True) -> CameraSnapshot:
        frame = self._read_frame()
        if frame is None:
            frame = self._placeholder("waiting for video")
        if copy:
            frame = frame.copy()
        return CameraSnapshot(
            frames={self.link.ip: frame},
            frame_versions={self.link.ip: self.version},
            frame_timestamps={self.link.ip: self.last_frame_at or time.monotonic()},
        )

    @property
    def running(self) -> bool:
        return self._running

    def _wait_for_first_frame(self) -> None:
        deadline = time.monotonic() + self.frame_timeout
        while time.monotonic() < deadline:
            frame = self._read_frame()
            if frame is not None:
                return
            time.sleep(0.03)
        raise TimeoutError(f"No video frame received from {self.link.ip} after {self.frame_timeout:.1f}s")

    def _read_frame(self) -> np.ndarray | None:
        frame_read = self.link.frame_read
        if frame_read is None:
            return None
        frame_rgb = getattr(frame_read, "frame", None)
        if frame_rgb is None:
            return None
        frame = np.asarray(frame_rgb)
        if frame.size == 0:
            return None
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.version += 1
        self.last_frame_at = time.monotonic()
        self._last_frame = frame_bgr
        return frame_bgr

    @staticmethod
    def _placeholder(message: str) -> np.ndarray:
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        frame[:, :] = (22, 25, 28)
        cv2.putText(frame, message, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 190, 190), 1, cv2.LINE_AA)
        return frame
